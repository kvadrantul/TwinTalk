"""
Extracts deep memories from a full chat history using LLM.

Produces a structured "memory document" (досье) about the relationship
between two chat participants by analyzing their entire message history.
"""

import json
import logging
import re
from typing import Callable, Optional

import openai

from config import WAVESPEED_API_KEY, WAVESPEED_BASE_URL, WAVESPEED_MODEL

logger = logging.getLogger(__name__)

# Standalone client — avoids importing from ai.client to prevent circular deps.
_client = openai.AsyncOpenAI(base_url=WAVESPEED_BASE_URL, api_key=WAVESPEED_API_KEY)
_model = WAVESPEED_MODEL

_CHUNK_SIZE = 200
_MAX_MESSAGES = 2000  # Sample at most this many messages for analysis
_TEMPERATURE = 0.3
_MAX_TOKENS = 4000

_EMPTY_RESULT: dict = {
    "relationship": "",
    "topics": [],
    "facts_about_each_other": [],
    "inside_jokes": [],
    "recurring_situations": [],
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Проанализируй переписку между {participant_name} и {other_name}.

Выпиши:
1. Краткое описание их отношений (1-2 предложения)
2. Ключевые темы обсуждений (до 20 тем) — для каждой: название, краткое описание, 1-2 конкретных примера сообщений
3. Факты которые они знают друг о друге (до 15 фактов)
4. Внутренние шутки и мемы (до 10)
5. Повторяющиеся ситуации/споры (до 10)

ВАЖНО: используй ТОЛЬКО факты из переписки. Не выдумывай.
Ответь СТРОГО в формате JSON:
{{
  "relationship": "...",
  "topics": [{{"theme": "...", "summary": "...", "key_messages": ["...", "..."]}}],
  "facts_about_each_other": ["..."],
  "inside_jokes": ["..."],
  "recurring_situations": ["..."]
}}

Переписка:
{chat_text}
"""

_MERGE_PROMPT = """\
Объедини эти промежуточные результаты анализа переписки в один консолидированный документ.
Убери дубликаты, объедини похожие темы. Лимиты: до 20 тем, до 15 фактов, до 10 шуток, до 10 ситуаций.
Ответь СТРОГО в том же JSON формате.

Промежуточные результаты:
{chunks_json}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_messages(messages: list) -> str:
    """Format a list of ParsedMessage into a readable chat log."""
    lines: list[str] = []
    for msg in messages:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{ts}] {msg.sender_name}: {msg.text}")
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """
    Attempt to parse a JSON object from the LLM response.

    Handles cases where the model wraps JSON in markdown code blocks.
    Returns _EMPTY_RESULT copy on failure.
    """
    cleaned = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        logger.warning("LLM response parsed as %s, expected dict", type(result).__name__)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)

    return dict(_EMPTY_RESULT)


async def _llm_call(prompt: str) -> str:
    """
    Single LLM call with one retry on failure.

    Returns the raw text content of the response.
    Raises only if both attempts fail.
    """
    last_error: Exception | None = None

    for attempt in range(1, 3):  # up to 2 attempts
        try:
            response = await _client.chat.completions.create(
                model=_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return content
            raise ValueError("Empty response from LLM")
        except Exception as exc:
            last_error = exc
            logger.warning(
                "LLM call attempt %d/2 failed: %s — retrying",
                attempt,
                exc,
            )
            if attempt < 2:
                import asyncio
                await asyncio.sleep(2)

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_memories(
    participant_name: str,
    other_name: str,
    messages: list,  # list of ParsedMessage
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    LLM reads the ENTIRE chat history and creates a structured memory document.

    For chats <= 200 messages: single LLM call.
    For chats > 200 messages: split into chunks, analyze each, then merge.

    Returns a dict with structure:
    {
        "relationship": "summary of their relationship",
        "topics": [
            {"theme": "topic name", "summary": "what happened", "key_messages": ["msg1", "msg2"]},
            ...
        ],
        "facts_about_each_other": [
            "fact 1",
            "fact 2"
        ],
        "inside_jokes": ["joke 1", "joke 2"],
        "recurring_situations": ["situation 1", "situation 2"]
    }

    progress_callback: optional async callable(message: str) for progress updates.
    Never raises — always returns at least a minimal valid dict.
    """
    try:
        # Sample messages if the chat is very large (to avoid hundreds of API calls)
        if len(messages) > _MAX_MESSAGES:
            import random
            # Stratified sample: take messages evenly spread across the timeline
            step = len(messages) // _MAX_MESSAGES
            sampled = messages[::step][:_MAX_MESSAGES]
            logger.info(
                "Sampled %d messages from %d total for memory extraction",
                len(sampled), len(messages),
            )
            messages = sampled

        if len(messages) <= _CHUNK_SIZE:
            return await _extract_single(
                participant_name, other_name, messages, progress_callback
            )
        return await _extract_chunked(
            participant_name, other_name, messages, progress_callback
        )
    except Exception as exc:
        logger.error("extract_memories failed entirely: %s", exc)
        return dict(_EMPTY_RESULT)


async def _extract_single(
    participant_name: str,
    other_name: str,
    messages: list,
    progress_callback: Optional[Callable],
) -> dict:
    """Extract memories from a small chat in a single LLM call."""
    if progress_callback:
        await progress_callback("Анализирую переписку…")

    chat_text = _format_messages(messages)
    prompt = _EXTRACTION_PROMPT.format(
        participant_name=participant_name,
        other_name=other_name,
        chat_text=chat_text,
    )

    raw = await _llm_call(prompt)
    result = _parse_json_response(raw)

    if progress_callback:
        await progress_callback("Память извлечена ✓")

    logger.info(
        "Extracted memories for %s↔%s: %d topics, %d facts",
        participant_name,
        other_name,
        len(result.get("topics", [])),
        len(result.get("facts_about_each_other", [])),
    )
    return result


async def _extract_chunked(
    participant_name: str,
    other_name: str,
    messages: list,
    progress_callback: Optional[Callable],
) -> dict:
    """Extract memories from a large chat by chunking, then merging."""
    chunks: list[list] = [
        messages[i : i + _CHUNK_SIZE] for i in range(0, len(messages), _CHUNK_SIZE)
    ]
    total = len(chunks)

    intermediate_results: list[dict] = []

    for idx, chunk in enumerate(chunks, start=1):
        if progress_callback:
            await progress_callback(
                f"Анализирую часть {idx}/{total}…"
            )

        chat_text = _format_messages(chunk)
        prompt = _EXTRACTION_PROMPT.format(
            participant_name=participant_name,
            other_name=other_name,
            chat_text=chat_text,
        )

        try:
            raw = await _llm_call(prompt)
            intermediate_results.append(_parse_json_response(raw))
        except Exception as exc:
            logger.warning("Chunk %d/%d failed: %s — skipping", idx, total, exc)

    if not intermediate_results:
        logger.error("All chunks failed — returning empty result")
        return dict(_EMPTY_RESULT)

    # Merge phase
    if progress_callback:
        await progress_callback("Объединяю результаты…")

    chunks_json = json.dumps(intermediate_results, ensure_ascii=False, indent=2)
    merge_prompt = _MERGE_PROMPT.format(chunks_json=chunks_json)

    try:
        raw = await _llm_call(merge_prompt)
        result = _parse_json_response(raw)
    except Exception as exc:
        logger.error("Merge LLM call failed: %s — returning first chunk result", exc)
        result = intermediate_results[0]

    if progress_callback:
        await progress_callback("Память извлечена ✓")

    logger.info(
        "Extracted merged memories for %s↔%s from %d chunks: %d topics, %d facts",
        participant_name,
        other_name,
        total,
        len(result.get("topics", [])),
        len(result.get("facts_about_each_other", [])),
    )
    return result
