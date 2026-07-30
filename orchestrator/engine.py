import asyncio
import logging
import random
import re
import uuid
from typing import Optional

from db.repository import (
    get_characters_by_session,
    get_last_messages,
    search_original_messages,
    search_chat_history_fts,
    add_message,
    update_session_status,
)
from ai.client import WaveSpeedClient
from parser.profiler import dict_to_profile
from telegram_proxy.sender import TelegramBotProxy, TelegramAPIError
from orchestrator.timing import TimingEngine

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Core engine that manages the turn-by-turn conversation between two character bots.

    State machine: IDLE -> READY -> RUNNING -> PAUSED -> STOPPED
    Each session runs as an independent asyncio task.
    """

    def __init__(self, session_id: str, group_chat_id: int):
        self.session_id = session_id
        self.group_chat_id = group_chat_id
        self.status = "idle"  # idle, ready, running, paused, stopped

        # Loaded from DB during initialize()
        self.characters: list[dict] = []
        self.character_proxies: list[TelegramBotProxy] = []
        self.timing_engines: list[TimingEngine] = []

        self.ai_client = WaveSpeedClient()
        self._task: Optional[asyncio.Task] = None
        self._current_turn = 0
        self._next_speaker_index = 0  # 0 or 1
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused initially
        self._turns_since_last_recall = 0
        self._turns_on_current_topic = 0
        self._current_memory_category = None  # track which memory category is currently being used
        self._used_topics: set = set()  # track recently used topic indices to avoid repetition
        self._next_turn_speaker = 0  # track who starts (0 = char A, 1 = char B)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self):
        """
        Load session data from DB:
        1. Get characters for this session (returns 2 dicts with profile_json, few_shot_examples, token, name, id)
        2. Create TelegramBotProxy for each character
        3. Build TimingEngine from response time distributions in profiles
        4. Get last messages to determine who speaks next
        5. Set status to 'ready'
        """
        self.characters = await get_characters_by_session(self.session_id)
        if len(self.characters) != 2:
            raise ValueError(
                f"Session {self.session_id} has {len(self.characters)} characters, expected 2"
            )

        # Create bot proxies and timing engines for each character
        for char in self.characters:
            self.character_proxies.append(TelegramBotProxy(char["token"]))

            # Extract response times from the profile for the timing engine
            profile_data = char["profile_json"]  # dict
            response_times = self._extract_response_times(profile_data)
            self.timing_engines.append(TimingEngine(response_times))

        # Determine who speaks next from last messages
        last_msgs = await get_last_messages(self.session_id, count=1)
        if last_msgs:
            last_char_id = last_msgs[0]["character_id"]
            # If the last message was from character[0], next speaker is character[1]
            if last_char_id == self.characters[0]["id"]:
                self._next_speaker_index = 1
            else:
                self._next_speaker_index = 0
            self._current_turn = last_msgs[0]["turn_number"]
        else:
            self._next_speaker_index = 0
            self._current_turn = 0

        self.status = "ready"
        logger.info(
            "Orchestrator initialized for session %s, next speaker: %s",
            self.session_id,
            self.characters[self._next_speaker_index]["name"],
        )

    async def start(self):
        """Start the conversation loop as an asyncio task."""
        if self.status not in ("ready", "paused"):
            raise RuntimeError(f"Cannot start from status '{self.status}'")
        self.status = "running"
        self._pause_event.set()
        await update_session_status(self.session_id, "running")
        self._task = asyncio.create_task(self._conversation_loop())
        logger.info("Conversation started for session %s", self.session_id)

    async def pause(self):
        """Pause the conversation. The loop will wait at the next iteration."""
        self.status = "paused"
        self._pause_event.clear()
        await update_session_status(self.session_id, "paused")
        logger.info("Conversation paused for session %s", self.session_id)

    async def resume(self):
        """Resume a paused conversation."""
        self.status = "running"
        self._pause_event.set()
        await update_session_status(self.session_id, "running")
        logger.info("Conversation resumed for session %s", self.session_id)

    async def stop(self):
        """Stop the conversation and clean up."""
        self.status = "stopped"
        self._pause_event.set()  # unblock if paused
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Close all proxy sessions
        for proxy in self.character_proxies:
            await proxy.close()
        await update_session_status(self.session_id, "stopped")
        logger.info("Conversation stopped for session %s", self.session_id)

    async def set_speed(self, multiplier: float):
        """Update speed multiplier for both timing engines."""
        for engine in self.timing_engines:
            engine.set_speed_multiplier(multiplier)
        logger.info("Speed set to %.1fx for session %s", multiplier, self.session_id)

    # ── Internal ──────────────────────────────────────────────────────────

    async def _conversation_loop(self):
        """
        Main conversation loop.
        Catches all exceptions to prevent silent death of the task.
        """
        try:
            while self.status not in ("stopped",):
                # Wait if paused
                await self._pause_event.wait()

                speaker_idx = self._determine_next_speaker()

                try:
                    await self._generate_and_send(speaker_idx)
                except TelegramAPIError as exc:
                    # Bot removed from group or token issue — fatal
                    logger.error(
                        "Telegram API error for session %s, stopping: %s",
                        self.session_id,
                        exc,
                    )
                    self.status = "stopped"
                    for proxy in self.character_proxies:
                        await proxy.close()
                    await update_session_status(self.session_id, "stopped")
                    return
                except Exception as exc:
                    # WaveSpeed or other error — skip this turn
                    logger.warning(
                        "Error generating/sending message for session %s, skipping turn: %s",
                        self.session_id,
                        exc,
                    )
                    # Still advance the turn so we don't get stuck
                    self._current_turn += 1
                    self._next_speaker_index = 1 - speaker_idx
                    continue

                # Calculate delay using the speaker's timing engine
                delay = self.timing_engines[speaker_idx].get_next_delay()
                logger.info(
                    "Session %s: next delay %.1fs",
                    self.session_id,
                    delay,
                )
                await asyncio.sleep(delay)

                # Advance turn
                self._current_turn += 1
                self._next_speaker_index = 1 - speaker_idx

        except asyncio.CancelledError:
            logger.info("Conversation loop cancelled for session %s", self.session_id)
            raise
        except Exception as exc:
            # Last-resort catch to prevent silent task death
            logger.exception(
                "Unexpected error in conversation loop for session %s: %s",
                self.session_id,
                exc,
            )
            self.status = "stopped"
            await update_session_status(self.session_id, "stopped")

    def _determine_next_speaker(self) -> int:
        """
        Determine who speaks next.
        Default: alternate between speakers.
        """
        return self._next_speaker_index

    async def _generate_and_send(self, speaker_index: int):
        """
        Generate a message for the speaker and send it.

        1. Get speaker character data (name, profile_json, few_shot_examples, token, id)
        2. Get other character name
        3. Get last 15 messages from DB as conversation history
           Format: [{"sender": "name", "text": "message"}, ...]
        4. Call ai_client.generate_message(...)
        5. Send via character proxy
        6. Save to DB with uuid message id
        7. Log the message
        """
        speaker = self.characters[speaker_index]
        other = self.characters[1 - speaker_index]

        speaker_name: str = speaker["name"]
        other_name: str = other["name"]
        profile: dict = speaker["profile_json"]
        few_shot_examples: list[dict] = speaker["few_shot_examples"]
        proxy: TelegramBotProxy = self.character_proxies[speaker_index]

        # Build conversation history from last 15 messages (for conversational continuity)
        last_msgs = await get_last_messages(self.session_id, count=15)
        conversation_history: list[dict] = []
        for msg in last_msgs:
            sender_name = msg.get("sender_name") or self._char_id_to_name(msg["character_id"])
            conversation_history.append({
                "sender": sender_name,
                "text": msg["text"],
            })

        # Extract keywords from last 15 messages for FTS5 search
        original_context = []
        chat_context = []
        if conversation_history:
            recent_text = " ".join(m["text"] for m in conversation_history)
            stop_words = {"это", "что", "как", "так", "тоже", "уже", "ещё", "еще", "был", "была",
                          "были", "будет", "может", "просто", "вообще", "ну", "да", "нет"}
            words = re.findall(r'[а-яА-ЯёЁa-zA-Z]{4,}', recent_text)
            keywords = [w for w in set(w.lower() for w in words) if w not in stop_words]

            if keywords:
                query = " OR ".join(keywords[:10])

                # FTS5 search across ALL original messages
                original_context = await search_original_messages(self.session_id, query, limit=50)

                # FTS5 search across ALL generated chat history
                chat_context = await search_chat_history_fts(self.session_id, query, limit=50)

                logger.info("FTS5 search: %d keywords -> %d original, %d chat_history",
                             len(keywords), len(original_context), len(chat_context))

        # Combine FTS5 results with recent messages, deduplicate
        seen_texts = {m["text"] for m in conversation_history}
        combined_original = []

        for msg in chat_context:
            if msg["text"] not in seen_texts:
                combined_original.append(msg)
                seen_texts.add(msg["text"])

        for msg in original_context:
            if msg["text"] not in seen_texts:
                combined_original.append(msg)
                seen_texts.add(msg["text"])

        # Topic injection logic
        memory_hint = None

        if self._current_turn == 0:
            # First turn: always inject a topic to start the conversation
            memory_hint = self._pick_random_memory(speaker)
            if memory_hint:
                self._turns_on_current_topic = 0
                self._current_memory_category = self._detect_memory_category(memory_hint)
                logger.info("First turn topic for %s [%s]: %s", speaker_name, self._current_memory_category, memory_hint[:80])
        elif self._turns_on_current_topic >= 2:
            # After 2 turns on the same topic: deterministically switch
            memory_hint = self._pick_random_memory(speaker)
            if memory_hint:
                self._current_memory_category = self._detect_memory_category(memory_hint)
                logger.info("Topic switch for %s (after %d turns) [%s]: %s", speaker_name, self._turns_on_current_topic, self._current_memory_category, memory_hint[:80])
                self._turns_on_current_topic = 0
            else:
                self._turns_on_current_topic += 1
        else:
            # Continue on the same topic
            self._turns_on_current_topic += 1

        # Extract style_profile from speaker character data
        style_profile = speaker.get("style_analyzer") or speaker.get("style_analyzer_json") or {}

        # Generate message via AI
        text = await self.ai_client.generate_message(
            character_name=speaker_name,
            other_name=other_name,
            profile=profile,
            conversation_history=conversation_history,
            few_shot_examples=few_shot_examples,
            memories=speaker.get("memories_json"),
            memory_hint=memory_hint,
            original_history=combined_original,
            style_profile=style_profile,
        )

        # Send via Telegram bot proxy
        result = await proxy.send_message(self.group_chat_id, text)
        telegram_msg_id = result.get("result", {}).get("message_id")

        # Save to DB (store the original sender name so conversation history
        # always uses the correct name from the original chat)
        msg_id = str(uuid.uuid4())
        await add_message(
            msg_id=msg_id,
            session_id=self.session_id,
            character_id=speaker["id"],
            sender_name=speaker_name,
            text=text,
            turn_number=self._current_turn,
        )

        logger.info(
            "Turn %d | %s: %s (tg_msg_id=%s)",
            self._current_turn,
            speaker_name,
            text[:80],
            telegram_msg_id,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pick_random_memory(self, speaker: dict) -> Optional[str]:
        """Pick a random memory topic/joke/fact that hasn't been used recently."""
        memories = speaker.get("memories_json", {})
        if not memories:
            return None

        candidates = []

        # Collect topics
        for topic in memories.get("topics", []):
            theme = topic.get("theme", "")
            summary = topic.get("summary", "")
            if theme and theme not in self._used_topics:
                candidates.append(f"вы обсуждали: {theme} — {summary}")

        # Collect facts
        for fact in memories.get("facts_about_each_other", []):
            if fact not in self._used_topics:
                candidates.append(f"ты знаешь что: {fact}")

        # Collect jokes
        for joke in memories.get("inside_jokes", []):
            if joke not in self._used_topics:
                candidates.append(f"ваша внутренняя шутка: {joke}")

        # Collect recurring situations
        for situation in memories.get("recurring_situations", []):
            if situation not in self._used_topics:
                candidates.append(f"повторяющаяся ситуация: {situation}")

        if not candidates:
            # All used, reset the used set and try again
            self._used_topics.clear()
            return self._pick_random_memory(speaker)  # retry once after clearing

        chosen = random.choice(candidates)
        self._used_topics.add(chosen)

        # Keep used_topics from growing too large
        if len(self._used_topics) > 30:
            # Remove oldest entries (keep last 15)
            self._used_topics = set(list(self._used_topics)[-15:])

        return chosen

    @staticmethod
    def _detect_memory_category(memory_hint: str) -> str:
        """Detect the category of a memory hint based on its prefix."""
        if memory_hint.startswith("вы обсуждали:"):
            return "topic"
        elif memory_hint.startswith("ты знаешь что:"):
            return "fact"
        elif memory_hint.startswith("ваша внутренняя шутка:"):
            return "joke"
        elif memory_hint.startswith("повторяющаяся ситуация:"):
            return "situation"
        else:
            return "unknown"

    def _char_id_to_name(self, character_id: str) -> str:
        """Map a character_id to its name."""
        for char in self.characters:
            if char["id"] == character_id:
                return char["name"]
        return "Unknown"

    @staticmethod
    def _extract_response_times(profile_data: dict) -> list[float]:
        """
        Extract response time samples from a character profile dict.

        The profile contains percentile stats (p25, p50, p75, p90) and average.
        We reconstruct a synthetic distribution from these for sampling.
        """
        times: list[float] = []

        avg = profile_data.get("avg_response_time_seconds", 0)
        p25 = profile_data.get("response_time_p25", 0)
        p50 = profile_data.get("response_time_p50", 0)
        p75 = profile_data.get("response_time_p75", 0)
        p90 = profile_data.get("response_time_p90", 0)

        # If we have percentile data, build a synthetic distribution
        if p50 > 0:
            # Generate ~20 synthetic samples spread across the distribution
            # Cluster more around the median for realism
            times.extend([max(1.0, p25)] * 3)
            times.extend([max(1.0, p50)] * 5)
            times.extend([max(1.0, p75)] * 3)
            times.extend([max(1.0, p90)] * 2)
            if avg > 0:
                times.append(max(1.0, avg))
        elif avg > 0:
            # Fallback: just use the average
            times = [avg]

        return times
