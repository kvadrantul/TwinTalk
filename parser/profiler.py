"""
Character Profiler — analyzes a person's chat history and extracts a
"character profile" describing their messaging style.  The profile is
used to instruct the AI (WaveSpeed) to generate messages in the same style.
"""

import re
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Optional

from .export_parser import ParsedMessage

# ---------------------------------------------------------------------------
# Stop-words (Russian + English) used when extracting n-grams
# ---------------------------------------------------------------------------
_STOP_WORDS: set[str] = {
    # Russian
    "и", "в", "на", "с", "по", "для", "к", "из", "о", "об", "а", "но",
    "что", "как", "это", "то", "не", "да", "ну", "же", "бы", "ли", "ни",
    "я", "ты", "он", "она", "оно", "мы", "вы", "они", "мне", "тебе",
    "себе", "все", "всё", "так", "уже", "ещё", "еще", "только", "если",
    "был", "была", "было", "были", "being", "есть", "нет",
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "when", "where",
    "how", "what", "which", "who", "whom", "this", "that", "these",
    "those", "i", "me", "my", "myself", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "what", "about", "up", "out", "then", "here", "there",
}

# Regex for emoji detection across common Unicode emoji blocks
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\U00002600-\U000026FF"   # misc symbols
    "]",
    flags=re.UNICODE,
)

# Common Russian greeting words (lowercased)
_GREETING_WORDS = [
    "привет", "здравствуй", "здравствуйте", "хай", "хей", "приветик",
    "доброе", "добрый", "добрая", "доброго", "дарова", "здарова",
    "йо", "хелло", "прив", "ку",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class CharacterProfile:
    """Complete messaging style profile of a person."""

    name: str
    avg_message_length: float
    median_message_length: float
    short_message_ratio: float
    long_message_ratio: float
    avg_words_per_message: float
    emoji_frequency: float
    top_emojis: list[str]
    emoji_description: str
    avg_response_time_seconds: float
    response_time_p25: float
    response_time_p50: float
    response_time_p75: float
    response_time_p90: float
    common_phrases: list[str]
    punctuation_habits: str
    exclamation_ratio: float
    question_ratio: float
    no_punctuation_ratio: float
    style_description: str
    response_style: str
    greeting_patterns: list[str]
    message_splitting_ratio: float


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _extract_emojis(text: str) -> list[str]:
    """Extract all emoji characters from *text* using Unicode ranges."""
    return _EMOJI_RE.findall(text)


def _compute_percentiles(
    values: list[float], percentiles: list[float]
) -> dict[str, float]:
    """Compute percentile values from a list of numbers.

    Returns a dict mapping each percentile label (e.g. ``"p25"``) to the
    computed value.  Returns 0.0 for every requested percentile when
    *values* is empty.
    """
    if not values:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    sorted_vals = sorted(values)
    result: dict[str, float] = {}
    for p in percentiles:
        k = (p / 100.0) * (len(sorted_vals) - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            result[f"p{int(p)}"] = sorted_vals[int(k)]
        else:
            result[f"p{int(p)}"] = sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
    return result


def _extract_ngrams(text: str, n: int) -> list[str]:
    """Extract word-level n-grams from *text*."""
    words = text.lower().split()
    # Strip pure-punctuation tokens
    words = [w.strip(".,!?;:()[]{}\"'—–-") for w in words]
    words = [w for w in words if w and w not in _STOP_WORDS]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _generate_emoji_description(top_emojis: list[str], frequency: float) -> str:
    """Generate a Russian-language emoji usage description."""
    if not top_emojis:
        return "эмодзи не использует"
    emoji_str = "".join(top_emojis[:5])
    if frequency < 0.1:
        return f"редко использует эмодзи, иногда {emoji_str}"
    if frequency < 0.4:
        return f"иногда использует эмодзи: {emoji_str}"
    if frequency < 0.8:
        return f"часто использует эмодзи: {emoji_str}"
    return f"очень часто использует эмодзи: {emoji_str}"


def _generate_response_style(
    avg_response_time: float, avg_length: float, split_ratio: float
) -> str:
    """Generate a Russian-language response style description."""
    parts: list[str] = []

    # Speed
    if avg_response_time < 30:
        parts.append("отвечает очень быстро")
    elif avg_response_time < 120:
        parts.append("отвечает быстро")
    elif avg_response_time < 600:
        parts.append("отвечает с умеренной скоростью")
    elif avg_response_time < 3600:
        parts.append("отвечает неспешно")
    else:
        parts.append("отвечает с большой задержкой")

    # Length
    if avg_length < 20:
        parts.append("короткими сообщениями")
    elif avg_length < 60:
        parts.append("сообщениями средней длины")
    elif avg_length < 150:
        parts.append("довольно длинными сообщениями")
    else:
        parts.append("длинными сообщениями")

    # Splitting
    if split_ratio > 0.5:
        parts.append("часто разбивает на несколько коротких сообщений")
    elif split_ratio > 0.25:
        parts.append("иногда разбивает мысли на несколько сообщений")

    return ", ".join(parts)


def _generate_style_description(profile_data: dict) -> str:
    """Generate an overall Russian-language style description from profile metrics."""
    parts: list[str] = []

    # Length tendencies
    avg_len = profile_data.get("avg_message_length", 0)
    if avg_len < 20:
        parts.append("пишет очень коротко")
    elif avg_len < 50:
        parts.append("пишет кратко")
    elif avg_len < 120:
        parts.append("пишет развернуто")
    else:
        parts.append("пишет длинными сообщениями")

    # Punctuation
    excl = profile_data.get("exclamation_ratio", 0)
    quest = profile_data.get("question_ratio", 0)
    no_punct = profile_data.get("no_punctuation_ratio", 0)

    punct_parts: list[str] = []
    if no_punct > 0.5:
        punct_parts.append("часто без пунктуации")
    if excl > 0.15:
        punct_parts.append("использует !")
    if quest > 0.15:
        punct_parts.append("задаёт вопросы")
    if punct_parts:
        parts.append(", ".join(punct_parts))

    # Emoji
    emoji_freq = profile_data.get("emoji_frequency", 0)
    if emoji_freq > 0.5:
        parts.append("активно использует эмодзи")
    elif emoji_freq > 0.15:
        parts.append("иногда использует эмодзи")

    # Response style
    resp_time = profile_data.get("avg_response_time_seconds", 0)
    if resp_time < 60:
        parts.append("отвечает быстро")
    elif resp_time < 600:
        parts.append("отвечает не торопясь")
    elif resp_time < 3600:
        parts.append("отвечает медленно")

    # Splitting
    split = profile_data.get("message_splitting_ratio", 0)
    if split > 0.4:
        parts.append("дробит сообщения")

    return "; ".join(parts) if parts else "обычный стиль переписки"


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------
def build_character_profile(
    name: str,
    messages: list[ParsedMessage],
    other_messages: list[ParsedMessage],
) -> CharacterProfile:
    """Analyze messages and build a complete character profile.

    Parameters
    ----------
    name:
        Display name of the person being profiled.
    messages:
        All :class:`ParsedMessage` objects sent by *this* person.
    other_messages:
        All :class:`ParsedMessage` objects sent by the *other* person.
    """

    # -- edge case: no messages at all --
    if not messages:
        return CharacterProfile(
            name=name,
            avg_message_length=0,
            median_message_length=0,
            short_message_ratio=0,
            long_message_ratio=0,
            avg_words_per_message=0,
            emoji_frequency=0,
            top_emojis=[],
            emoji_description="эмодзи не использует",
            avg_response_time_seconds=0,
            response_time_p25=0,
            response_time_p50=0,
            response_time_p75=0,
            response_time_p90=0,
            common_phrases=[],
            punctuation_habits="нет данных",
            exclamation_ratio=0,
            question_ratio=0,
            no_punctuation_ratio=0,
            style_description="нет данных",
            response_style="нет данных",
            greeting_patterns=[],
            message_splitting_ratio=0,
        )

    texts = [m.text for m in messages]
    lengths = [len(t) for t in texts]
    n = len(texts)

    # 1. Message length analysis -------------------------------------------
    avg_len = statistics.mean(lengths)
    med_len = statistics.median(lengths)
    short_ratio = sum(1 for l in lengths if l < 20) / n
    long_ratio = sum(1 for l in lengths if l > 100) / n

    word_counts = [len(t.split()) for t in texts]
    avg_words = statistics.mean(word_counts)

    # 2. Emoji analysis ----------------------------------------------------
    all_emojis: list[str] = []
    for t in texts:
        all_emojis.extend(_extract_emojis(t))
    emoji_freq = len(all_emojis) / n if n else 0
    emoji_counter = Counter(all_emojis)
    top_emojis = [e for e, _ in emoji_counter.most_common(10)]
    emoji_desc = _generate_emoji_description(top_emojis, emoji_freq)

    # 3. Response time analysis --------------------------------------------
    # Merge & sort all messages by timestamp
    all_msgs = sorted(messages + other_messages, key=lambda m: m.timestamp)
    this_senders = {m.sender_name for m in messages}
    other_senders = {m.sender_name for m in other_messages}

    response_times: list[float] = []
    last_other_ts: Optional[float] = None
    for msg in all_msgs:
        if msg.sender_name in other_senders:
            last_other_ts = msg.timestamp.timestamp()
        elif msg.sender_name in this_senders and last_other_ts is not None:
            rt = msg.timestamp.timestamp() - last_other_ts
            if rt >= 0:
                response_times.append(rt)

    pcts = _compute_percentiles(response_times, [25, 50, 75, 90])
    avg_rt = statistics.mean(response_times) if response_times else 0

    # 4. Common phrases (bigrams + trigrams) --------------------------------
    ngram_counter: Counter = Counter()
    for t in texts:
        ngram_counter.update(_extract_ngrams(t, 2))
        ngram_counter.update(_extract_ngrams(t, 3))
    common_phrases = [ng for ng, _ in ngram_counter.most_common(10)]

    # 5. Punctuation analysis -----------------------------------------------
    excl_count = sum(1 for t in texts if t.rstrip().endswith("!"))
    quest_count = sum(1 for t in texts if t.rstrip().endswith("?"))
    no_punct_count = sum(
        1 for t in texts if t.rstrip() and not t.rstrip()[-1] in ".!?…,"
    )
    excl_ratio = excl_count / n
    quest_ratio = quest_count / n
    no_punct_ratio = no_punct_count / n

    # Build punctuation habits string
    punct_parts: list[str] = []
    if no_punct_ratio > 0.5:
        punct_parts.append("чаще без точки")
    elif no_punct_ratio > 0.25:
        punct_parts.append("иногда без точки")
    if excl_ratio > 0.15:
        punct_parts.append("использует !")
    if quest_ratio > 0.15:
        punct_parts.append("использует ?")
    if not punct_parts:
        punct_parts.append("стандартная пунктуация")
    punctuation_habits = ", ".join(punct_parts)

    # 6. Message splitting --------------------------------------------------
    sorted_own = sorted(messages, key=lambda m: m.timestamp)
    split_count = 0
    for i in range(1, len(sorted_own)):
        delta = (sorted_own[i].timestamp - sorted_own[i - 1].timestamp).total_seconds()
        if delta <= 5:
            split_count += 1
    split_ratio = split_count / (n - 1) if n > 1 else 0

    # 7. Greeting patterns --------------------------------------------------
    greeting_counter: Counter = Counter()
    for t in texts:
        lower = t.strip().lower()
        for gw in _GREETING_WORDS:
            if lower.startswith(gw):
                # Grab the first few words as the pattern
                first_words = " ".join(lower.split()[:3])
                greeting_counter[first_words] += 1
                break
    greeting_patterns = [g for g, _ in greeting_counter.most_common(5)]

    # 8. Human-readable descriptions ----------------------------------------
    profile_data = {
        "avg_message_length": avg_len,
        "exclamation_ratio": excl_ratio,
        "question_ratio": quest_ratio,
        "no_punctuation_ratio": no_punct_ratio,
        "emoji_frequency": emoji_freq,
        "avg_response_time_seconds": avg_rt,
        "message_splitting_ratio": split_ratio,
    }
    style_desc = _generate_style_description(profile_data)
    resp_style = _generate_response_style(avg_rt, avg_len, split_ratio)

    return CharacterProfile(
        name=name,
        avg_message_length=round(avg_len, 2),
        median_message_length=round(med_len, 2),
        short_message_ratio=round(short_ratio, 4),
        long_message_ratio=round(long_ratio, 4),
        avg_words_per_message=round(avg_words, 2),
        emoji_frequency=round(emoji_freq, 4),
        top_emojis=top_emojis,
        emoji_description=emoji_desc,
        avg_response_time_seconds=round(avg_rt, 2),
        response_time_p25=round(pcts["p25"], 2),
        response_time_p50=round(pcts["p50"], 2),
        response_time_p75=round(pcts["p75"], 2),
        response_time_p90=round(pcts["p90"], 2),
        common_phrases=common_phrases,
        punctuation_habits=punctuation_habits,
        exclamation_ratio=round(excl_ratio, 4),
        question_ratio=round(quest_ratio, 4),
        no_punctuation_ratio=round(no_punct_ratio, 4),
        style_description=style_desc,
        response_style=resp_style,
        greeting_patterns=greeting_patterns,
        message_splitting_ratio=round(split_ratio, 4),
    )


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------
def select_few_shot_examples(
    messages: list[ParsedMessage],
    count: int = 15,
) -> list[dict]:
    """Select diverse representative examples of a person's messages.

    Strategy:
    - Pick messages of varying lengths (short, medium, long)
    - Pick messages with emojis and without
    - Pick messages with different punctuation
    - Pick messages from different time periods
    - Return as list of ``{"sender": name, "text": message_text}``
    """
    if not messages:
        return []

    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
    name = sorted_msgs[0].sender_name

    # Categorise by length
    short = [m for m in sorted_msgs if len(m.text) < 20]
    medium = [m for m in sorted_msgs if 20 <= len(m.text) < 100]
    long_ = [m for m in sorted_msgs if len(m.text) >= 100]

    # Categorise by emoji
    with_emoji = [m for m in sorted_msgs if _extract_emojis(m.text)]
    without_emoji = [m for m in sorted_msgs if not _extract_emojis(m.text)]

    # Punctuation variants
    excl = [m for m in sorted_msgs if m.text.rstrip().endswith("!")]
    quest = [m for m in sorted_msgs if m.text.rstrip().endswith("?")]

    selected_ids: set[int] = set()
    selected: list[ParsedMessage] = []

    def _add(msgs_pool: list[ParsedMessage], n: int) -> None:
        """Add up to *n* diverse messages from *msgs_pool*."""
        if not msgs_pool:
            return
        # Spread across time: pick evenly spaced indices
        step = max(1, len(msgs_pool) // max(n, 1))
        picked = 0
        idx = 0
        while picked < n and idx < len(msgs_pool):
            m = msgs_pool[idx]
            if m.id not in selected_ids:
                selected.append(m)
                selected_ids.add(m.id)
                picked += 1
            idx += step
        # Fill remaining from the start if needed
        for m in msgs_pool:
            if picked >= n:
                break
            if m.id not in selected_ids:
                selected.append(m)
                selected_ids.add(m.id)
                picked += 1

    # Distribute budget across categories
    third = count // 3
    _add(short, third)
    _add(medium, third)
    _add(long_, count - len(selected))

    _add(with_emoji, max(1, count // 5))
    _add(excl, max(1, count // 6))
    _add(quest, max(1, count // 6))

    # If still short, fill from full timeline
    if len(selected) < count:
        _add(sorted_msgs, count - len(selected))

    # Sort final selection chronologically
    selected.sort(key=lambda m: m.timestamp)

    return [{"sender": name, "text": m.text} for m in selected[:count]]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def profile_to_dict(profile: CharacterProfile) -> dict:
    """Convert profile to a JSON-serializable dict."""
    return asdict(profile)


def dict_to_profile(data: dict) -> CharacterProfile:
    """Reconstruct a :class:`CharacterProfile` from a dict."""
    return CharacterProfile(**data)
