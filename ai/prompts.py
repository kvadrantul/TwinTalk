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


def build_system_prompt(character_name: str, other_name: str, profile: dict, memories: dict = None, style_profile: dict = None) -> str:
    """
    Build system prompt for message generation.

    style_profile dict (from style_analyzer) contains:
    - communication_style: str
    - typical_phrases: list[str]
    - personality_traits: list[str]
    - life_context: str
    - relationship_with_other: str
    - never_says: list[str]
    """
    prompt = f"Ты — {character_name}. Ты переписываешься в Telegram с {other_name}.\n"

    # Deep style profile from style_analyzer
    if style_profile:
        style = style_profile.get("communication_style", "")
        if style:
            prompt += f"\nТвой стиль общения:\n{style}\n"

        phrases = style_profile.get("typical_phrases", [])
        if phrases:
            prompt += f"\nТипичные фразы: {', '.join('\"' + p + '\"' for p in phrases[:15])}\n"

        traits = style_profile.get("personality_traits", [])
        if traits:
            prompt += f"Черты характера: {', '.join(traits)}\n"

        life = style_profile.get("life_context", "")
        if life:
            prompt += f"\nКонтекст жизни: {life}\n"

        relationship = style_profile.get("relationship_with_other", "")
        if relationship:
            prompt += f"Отношения с {other_name}: {relationship}\n"

        never = style_profile.get("never_says", [])
        if never:
            prompt += f"\nНИКОГДА не говори: {', '.join(never[:10])}\n"

    # Memories (topics, facts, jokes)
    if memories:
        memory_block = build_memory_block(memories)
        if memory_block:
            prompt += f"\n{memory_block}\n"

    prompt += (
        f"\nПРАВИЛА:\n"
        f"Пиши ТОЛЬКО текст сообщения. Без пояснений, без кавычек, без markdown.\n"
        f"Копируй свой стиль из примеров выше.\n"
        f"Отвечай только на основе того что реально было в вашей переписке.\n"
        f"Не выдумывай события которых не было.\n"
        f"Можешь вспомнить что-то из прошлого, пошутить внутреннюю шутку, спросить о чём-то."
    )

    return prompt


def build_conversation_messages(
    conversation_history: list[dict],
    character_name: str,
    other_name: str,
    few_shot_examples: list[dict],
    original_history: list[dict] = None,
) -> list[dict]:
    """
    Build the messages array for the OpenAI chat completion API call.

    Structure:
    1. System message (from build_system_prompt)
    2. (Optional) Original chat history as "past memories" block
    3. Few-shot examples as alternating user/assistant messages
    4. Last 50 conversation messages as context

    Each message in history has: {"sender": "name", "text": "message text"}
    """
    # Determine profile from context (passed separately to the caller)
    # Here we just build the message list; system prompt is added by the caller.
    messages: list[dict] = []

    # Inject original chat history as "past memories" before few-shot examples
    if original_history:
        memory_lines = []
        for msg in original_history:
            sender = msg.get("sender_name", msg.get("sender", ""))
            text = msg.get("text", "")
            if sender and text:
                memory_lines.append(f"{sender}: {text}")

        if memory_lines:
            memory_block = "\n".join(memory_lines[-30:])  # last 30 from original
            messages.append({
                "role": "system",
                "content": f"Вот фрагменты ваших реальных прошлых разговоров, которые ты вспоминаешь:\n\n{memory_block}"
            })

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
