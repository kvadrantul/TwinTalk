import asyncio
import logging
import random
import re
import uuid
from typing import Optional

from db.repository import (
    get_characters_by_session,
    get_last_messages,
    get_original_messages,
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
        self._current_seed: Optional[str] = None
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
        3. Pick a random seed message from full original history (>5 words)
        4. FTS5-search by seed keywords for relevant context
        5. Get last 10 messages as conversation_history (for continuity)
        6. Call ai_client.generate_message(...)
        7. Send via character proxy
        8. Save to DB with uuid message id
        9. Log the message
        """
        speaker = self.characters[speaker_index]
        other = self.characters[1 - speaker_index]

        speaker_name: str = speaker["name"]
        other_name: str = other["name"]
        profile: dict = speaker["profile_json"]
        few_shot_examples: list[dict] = speaker["few_shot_examples"]
        proxy: TelegramBotProxy = self.character_proxies[speaker_index]

        # ── 1. Pick seed ──────────────────────────────────────────────
        # Decide whether to pick a new seed this turn
        need_new_seed = (
            self._current_turn == 0
            or self._turns_on_current_topic >= 4
        )

        seed_text = None
        if need_new_seed:
            all_msgs = await get_original_messages(self.session_id)
            # Filter to messages with >5 words
            long_msgs = [m for m in all_msgs if len(m.get("text", "").split()) >= 5]
            if long_msgs:
                # Try up to 10 times to find one with >10 words
                seed_msg = None
                for _ in range(10):
                    candidate = random.choice(long_msgs)
                    if len(candidate.get("text", "").split()) > 10:
                        seed_msg = candidate
                        break
                if seed_msg is None:
                    seed_msg = random.choice(long_msgs)
                seed_text = seed_msg.get("text", "")
                logger.info(
                    "Seed selected for session %s (%d words): %s",
                    self.session_id,
                    len(seed_text.split()),
                    seed_text[:120],
                )
            self._turns_on_current_topic = 0
        else:
            # Reuse previous seed (stored on the instance)
            seed_text = self._current_seed
            self._turns_on_current_topic += 1

        # Persist seed for reuse on non-switch turns
        self._current_seed = seed_text

        # ── 2. FTS5 search by seed keywords ───────────────────────────
        original_context = []
        chat_context = []
        if seed_text:
            stop_words = {"это", "что", "как", "так", "тоже", "уже", "ещё", "еще", "был", "была",
                          "были", "будет", "может", "просто", "вообще", "ну", "да", "нет"}
            words = re.findall(r'[а-яА-ЯёЁa-zA-Z]{4,}', seed_text)
            keywords = [w for w in set(w.lower() for w in words) if w not in stop_words]

            if keywords:
                query = " OR ".join(keywords[:10])

                # FTS5 search across ALL original messages
                original_context = await search_original_messages(self.session_id, query, limit=20)

                # FTS5 search across ALL generated chat history
                chat_context = await search_chat_history_fts(self.session_id, query, limit=15)

                logger.info(
                    "FTS5 search from seed: %d keywords -> %d original, %d chat_history",
                    len(keywords), len(original_context), len(chat_context),
                )

        # ── 3. Build conversation history (last 10 messages) ──────────
        last_msgs = await get_last_messages(self.session_id, count=10)
        conversation_history: list[dict] = []
        for msg in last_msgs:
            sender_name = msg.get("sender_name") or self._char_id_to_name(msg["character_id"])
            conversation_history.append({
                "sender": sender_name,
                "text": msg["text"],
            })

        # Combine FTS5 results, filter out short messages (<5 words), deduplicate
        seen_texts = {m["text"] for m in conversation_history}
        combined_original = []

        for msg in chat_context:
            text = msg.get("text", "")
            if len(text.split()) >= 5 and text not in seen_texts:
                combined_original.append(msg)
                seen_texts.add(text)

        for msg in original_context:
            text = msg.get("text", "")
            if len(text.split()) >= 5 and text not in seen_texts:
                combined_original.append(msg)
                seen_texts.add(text)

        logger.info(
            "Context for session %s: seed='%s', combined_original=%d msgs, conversation_history=%d msgs",
            self.session_id,
            (seed_text or "")[:60],
            len(combined_original),
            len(conversation_history),
        )

        # memory_hint = the seed message text (topic of conversation)
        memory_hint = seed_text

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
