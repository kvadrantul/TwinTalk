"""
Deep style analysis using LLM. Reads the ENTIRE chat history and creates
a comprehensive style profile for a character.
"""

import json
import logging
import re
from typing import Optional

import openai
from config import WAVESPEED_API_KEY, WAVESPEED_BASE_URL, WAVESPEED_MODEL

logger = logging.getLogger(__name__)

_client = openai.AsyncOpenAI(base_url=WAVESPEED_BASE_URL, api_key=WAVESPEED_API_KEY)
_model = WAVESPEED_MODEL

_EMPTY_PROFILE = {
    "character": "",
    "communication_style": "",
    "typical_phrases": [],
    "attitude_towards_topics": {},
    "life_context": "",
    "relationship_with_other": "",
    "never_says": [],
    "personality_traits": [],
}


async def analyze_style(
    participant_name: str,
    other_name: str,
    messages: list,  # list of ParsedMessage - ALL messages from this person
    all_messages: list = None,  # ALL messages from both people (for context)
) -> dict:
    """
    LLM reads ALL messages and creates a deep style profile.

    The entire chat history is sent in ONE API call (claude-sonnet-5 has 1M context).

    Returns dict with:
    - character: who this person is
    - communication_style: how they communicate (formal/casual, length, tone)
    - typical_phrases: actual phrases they use often
    - attitude_towards_topics: what they think about various topics
    - life_context: where they live, work, hobbies (from chat)
    - relationship_with_other: how they relate to the other person
    - never_says: things this person would never say
    - personality_traits: key personality traits
    """
    try:
        if all_messages is None:
            all_messages = messages

        # Format ALL messages as a chat log
        chat_lines = []
        for msg in all_messages:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            chat_lines.append(f"[{ts}] {msg.sender_name}: {msg.text}")
        chat_text = "\n".join(chat_lines)

        prompt = f"""Проанализируй переписку между {participant_name} и {other_name}.

Создай ГЛУБОКИЙ профиль стиля общения {participant_name} на основе ВСЕЙ переписки.

Ответь СТРОГО в формате JSON:
{{
  "character": "Кто этот человек (краткое описание)",
  "communication_style": "Как он общается (длина сообщений, тон, формальность, сленг, конкретные примеры)",
  "typical_phrases": ["фраза1", "фраза2", ...],
  "attitude_towards_topics": {{
    "тема1": "отношение",
    "тема2": "отношение"
  }},
  "life_context": "Контекст жизни: город, работа, хобби, что известно из переписки",
  "relationship_with_other": "Как относится к {other_name}, формат общения, расстояние между ними",
  "never_says": ["вещь которую он никогда не скажет", ...],
  "personality_traits": ["черта1", "черта2", ...]
}}

ВАЖНО:
- Используй ТОЛЬКО факты из переписки
- typical_phrases должны быть РЕАЛЬНЫМИ фразами из переписки (копируй дословно)
- never_says — что этот человек НИКОГДА не сказал бы (исходя из его стиля)
- Будь максимально конкретным и детальным

Переписка:
{chat_text}"""

        logger.info("Analyzing style for %s from %d messages (full history)",
                     participant_name, len(all_messages))

        response = await _client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )

        content = response.choices[0].message.content or ""
        result = _parse_json_response(content)
        result["character"] = participant_name

        logger.info("Style analysis complete for %s: %d phrases, %d traits",
                     participant_name,
                     len(result.get("typical_phrases", [])),
                     len(result.get("personality_traits", [])))

        return result

    except Exception as exc:
        logger.error("Style analysis failed for %s: %s", participant_name, exc)
        result = dict(_EMPTY_PROFILE)
        result["character"] = participant_name
        return result


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse style analysis JSON: %s", exc)
    return dict(_EMPTY_PROFILE)
