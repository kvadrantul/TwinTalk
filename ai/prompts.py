def build_memory_block(memories: dict) -> str:
    """Convert memories dict into a readable text block for the system prompt."""
    if not memories or not memories.get("topics"):
        return ""

    parts = []

    # Relationship summary
    rel = memories.get("relationship", "")
    if rel:
        parts.append(f"Ваши отношения: {rel}")

    # Topics
    topics = memories.get("topics", [])
    if topics:
        topic_lines = []
        for t in topics[:15]:  # limit to 15 topics
            theme = t.get("theme", "")
            summary = t.get("summary", "")
            if theme and summary:
                topic_lines.append(f"  - {theme}: {summary}")
        if topic_lines:
            parts.append("Ключевые темы ваших разговоров:\n" + "\n".join(topic_lines))

    # Facts
    facts = memories.get("facts_about_each_other", [])
    if facts:
        parts.append("Факты которые вы знаете друг о друге:\n" + "\n".join(f"  - {f}" for f in facts[:10]))

    # Inside jokes
    jokes = memories.get("inside_jokes", [])
    if jokes:
        parts.append("Ваши внутренние шутки: " + ", ".join(f'"{j}"' for j in jokes[:8]))

    # Recurring situations
    situations = memories.get("recurring_situations", [])
    if situations:
        parts.append("Повторяющиеся ситуации: " + ", ".join(situations[:8]))

    if not parts:
        return ""

    return "\n\n".join(parts)


def build_system_prompt(character_name: str, other_name: str, profile: dict, memories: dict = None) -> str:
    """
    Build the system prompt for message generation.

    Profile dict contains:
    - avg_message_length: int
    - emoji_description: str (e.g., "часто использует 🔥😂❤️")
    - punctuation_habits: str (e.g., "чаще без точки в конце, иногда !")
    - common_phrases: list[str]
    - style_description: str (overall style summary)
    """
    avg_len = profile.get("avg_message_length", 50)
    emoji_desc = profile.get("emoji_description", "использует эмодзи умеренно")
    punctuation = profile.get("punctuation_habits", "стандартная пунктуация")
    common_phrases = profile.get("common_phrases", [])
    style_desc = profile.get("style_description", "обычный стиль общения")

    phrases_str = ""
    if common_phrases:
        phrases_str = "\n    - Частые выражения: " + ", ".join(f'"{p}"' for p in common_phrases)

    prompt = (
        f"Ты — {character_name}. Ты переписываешься в Telegram с {other_name}.\n"
        f"Генерируй СЛЕДУЮЩЕЕ сообщение, которое {character_name} отправил бы.\n"
        f"\n"
        f"Твой стиль общения:\n"
        f"    - Средняя длина сообщения: ~{avg_len} символов\n"
        f"    - Эмодзи: {emoji_desc}\n"
        f"    - Пунктуация: {punctuation}\n"
        f"    - Стиль: {style_desc}"
        f"{phrases_str}\n"
    )

    # Append memory block if provided
    if memories:
        memory_block = build_memory_block(memories)
        if memory_block:
            prompt += (
                f"\nПамять о ваших реальных разговорах:\n"
                f"{memory_block}\n"
                f"\n"
                f"Иногда ты можешь спонтанно вспомнить что-то из этого и поднять тему — так делают реальные люди.\n"
                f"Но не упоминай это каждый раз — только когда это естественно подходит.\n"
            )

    prompt += (
        f"\n"
        f"ПРАВИЛА:\n"
        f"    - Пиши ТОЛЬКО текст сообщения, без пояснений и кавычек\n"
        f"    - Без markdown, без форматирования\n"
        f"    - 1-3 предложения, как в реальном чате\n"
        f"    - Отвечай естественно на последнее сообщение собеседника\n"
        f"    - Используй свойственный тебе стиль и эмодзи"
    )

    return prompt


def build_conversation_messages(
    conversation_history: list[dict],
    character_name: str,
    other_name: str,
    few_shot_examples: list[dict],
) -> list[dict]:
    """
    Build the messages array for the OpenAI chat completion API call.

    Structure:
    1. System message (from build_system_prompt)
    2. Few-shot examples as alternating user/assistant messages
    3. Last 10-15 conversation messages as context

    Each message in history has: {"sender": "name", "text": "message text"}
    """
    # Determine profile from context (passed separately to the caller)
    # Here we just build the message list; system prompt is added by the caller.
    messages: list[dict] = []

    # Few-shot examples (limit to 10 to leave room for larger context)
    for example in few_shot_examples[:10]:
        sender = example.get("sender", "")
        text = example.get("text", "")
        if sender == other_name:
            messages.append({"role": "user", "content": text})
        elif sender == character_name:
            messages.append({"role": "assistant", "content": text})

    # Last 50 conversation messages for context
    recent = conversation_history[-50:]
    for msg in recent:
        sender = msg.get("sender", "")
        text = msg.get("text", "")
        if sender == other_name:
            messages.append({"role": "user", "content": text})
        elif sender == character_name:
            messages.append({"role": "assistant", "content": text})

    return messages
