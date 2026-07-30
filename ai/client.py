import asyncio
import logging
import re

import openai

import config
from ai.prompts import build_conversation_messages, build_system_prompt

logger = logging.getLogger(__name__)


class WaveSpeedClient:
    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(
            base_url=config.WAVESPEED_BASE_URL,
            api_key=config.WAVESPEED_API_KEY,
        )
        self.model = config.WAVESPEED_MODEL

    async def generate_message(
        self,
        character_name: str,
        other_name: str,
        profile: dict,
        conversation_history: list[dict],
        few_shot_examples: list[dict],
        temperature: float = 0.85,
        max_retries: int = 3,
        memories: dict = None,
        memory_hint: str = None,
        original_history: list[dict] = None,
        style_profile: dict = None,
    ) -> str:
        """
        Generate the next message for a character.

        1. Build system prompt using ai.prompts
        2. Build conversation messages array
        3. Call WaveSpeed API (chat.completions.create)
        4. Extract and validate the response text
        5. Clean up: strip quotes, markdown, extra whitespace
        6. Retry with exponential backoff on failure

        Returns: clean message text string
        Raises: Exception after max_retries exhausted
        """
        system_prompt = build_system_prompt(character_name, other_name, profile, memories=memories, style_profile=style_profile)

        if memory_hint:
            system_prompt += f"\n\n[Воспоминание: {memory_hint}]\nМожешь естественно упомянуть это если подходит, или игнорируй."

        conversation_messages = build_conversation_messages(
            conversation_history, character_name, other_name, few_shot_examples,
            original_history=original_history,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_messages,
        ]

        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "WaveSpeed API call attempt %d/%d for character '%s'",
                    attempt,
                    max_retries,
                    character_name,
                )
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

                content = response.choices[0].message.content or ""
                cleaned = self._clean_response(content)

                if not cleaned:
                    raise ValueError("Empty response after cleaning")

                logger.info(
                    "Generated message for '%s' (%d chars): %s",
                    character_name,
                    len(cleaned),
                    cleaned[:80],
                )
                return cleaned

            except Exception as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(
                    "WaveSpeed API attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                if attempt < max_retries:
                    await asyncio.sleep(wait)

        raise Exception(
            f"WaveSpeed API call failed after {max_retries} attempts"
        ) from last_error

    @staticmethod
    def _clean_response(text: str) -> str:
        """
        Clean the AI response:
        - Strip leading/trailing whitespace
        - Remove surrounding quotes if present
        - Remove markdown formatting (**bold**, _italic_, etc.)
        - Remove any prefix like "CharacterName: " if the model adds it
        - Truncate if absurdly long (>500 chars)
        """
        cleaned = text.strip()

        # Remove surrounding quotes (single, double, or smart quotes)
        if len(cleaned) >= 2:
            first, last = cleaned[0], cleaned[-1]
            if (first == last and first in ('"', "'", '"', "'")) or (
                first == '"' and last == '"'
            ):
                cleaned = cleaned[1:-1].strip()

        # Remove markdown bold/italic markers
        cleaned = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", cleaned)
        cleaned = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", cleaned)

        # Remove markdown code blocks / inline code
        cleaned = re.sub(r"`{1,3}[^`]*`{1,3}", "", cleaned)

        # Remove "Name: " prefix (e.g., "Анна: Привет!" → "Привет!")
        cleaned = re.sub(r"^[^:]{1,30}:\s+", "", cleaned)

        # Remove any remaining leading/trailing whitespace
        cleaned = cleaned.strip()

        # Truncate if absurdly long
        if len(cleaned) > 500:
            cleaned = cleaned[:500].rsplit(" ", 1)[0]

        return cleaned
