"""
Parses Telegram Desktop JSON exports (result.json) into structured data.

Handles both string and array-style text fields, filters service messages,
identifies participants, and validates the export format.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    """A single parsed chat message."""

    id: int
    sender_name: str
    sender_id: str
    text: str
    timestamp: datetime
    reply_to_message_id: Optional[int] = None


@dataclass
class ParsedChat:
    """A fully parsed Telegram chat export."""

    chat_name: str
    chat_type: str
    participants: list[str]  # names of the two participants
    messages: list[ParsedMessage]


def validate_export(data: dict) -> None:
    """
    Validate the JSON structure looks like a Telegram Desktop export.

    Raises:
        ValueError: if the data is missing required top-level keys.
    """
    if not isinstance(data, dict):
        raise ValueError("Export root must be a JSON object (dict).")
    if "messages" not in data:
        raise ValueError(
            "Invalid Telegram export: missing 'messages' key at top level."
        )
    if not isinstance(data["messages"], list):
        raise ValueError("Invalid Telegram export: 'messages' must be a list.")


def extract_text(text_field: Any) -> str:
    """
    Extract plain text from the ``text`` field of a Telegram message.

    The field can be:
    - A plain string → returned as-is.
    - A list of styled segments → all segment ``"text"`` values are concatenated.
    - Anything else (None, int, …) → converted via ``str()``.

    Returns:
        The concatenated plain-text string.
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: list[str] = []
        for segment in text_field:
            if isinstance(segment, dict) and "text" in segment:
                parts.append(str(segment["text"]))
            elif isinstance(segment, str):
                # Defensive: some exports may embed raw strings in the array.
                parts.append(segment)
        return "".join(parts)
    if text_field is None:
        return ""
    return str(text_field)


def identify_participants(messages: list[ParsedMessage]) -> list[str]:
    """
    Identify the two main participants from unique sender names.

    Args:
        messages: list of already-parsed messages (must be non-empty).

    Returns:
        A list of exactly 2 participant names.

    Raises:
        ValueError: if fewer or more than 2 unique senders are found.
    """
    seen: dict[str, None] = {}
    for msg in messages:
        if msg.sender_name not in seen:
            seen[msg.sender_name] = None
        if len(seen) > 2:
            # Keep scanning – we still want the full set to report in the error.
            pass

    unique_names = list(seen.keys())

    if len(unique_names) < 2:
        raise ValueError(
            f"Expected at least 2 participants, found {len(unique_names)}: "
            f"{unique_names}"
        )
    if len(unique_names) > 2:
        raise ValueError(
            f"Expected exactly 2 participants, found {len(unique_names)}: "
            f"{unique_names}. This parser is designed for 1-to-1 chats."
        )
    return unique_names


def _parse_single_message(raw: dict) -> Optional[ParsedMessage]:
    """
    Convert a raw JSON message object into a ParsedMessage.

    Returns None if the message should be skipped (service message, no sender,
    empty text, etc.).
    """
    # Only user-sent messages (not service actions).
    if raw.get("type") != "message":
        return None

    # Must have a sender.
    sender_name = raw.get("from")
    if not sender_name:
        return None

    sender_id = raw.get("from_id", "")

    # Extract and validate text.
    text = extract_text(raw.get("text"))
    if not text.strip():
        return None

    # Parse timestamp.
    date_str = raw.get("date", "")
    try:
        timestamp = datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        # Fall back to unixtime if ISO parsing fails.
        unixtime_str = raw.get("date_unixtime")
        if unixtime_str:
            try:
                timestamp = datetime.utcfromtimestamp(int(unixtime_str))
            except (ValueError, TypeError):
                logger.warning(
                    "Skipping message id=%s: unable to parse timestamp.", raw.get("id")
                )
                return None
        else:
            logger.warning(
                "Skipping message id=%s: unable to parse timestamp.", raw.get("id")
            )
            return None

    # Optional reply reference.
    reply_to = raw.get("reply_to_message_id")
    if reply_to is not None:
        try:
            reply_to = int(reply_to)
        except (ValueError, TypeError):
            reply_to = None

    return ParsedMessage(
        id=int(raw.get("id", 0)),
        sender_name=str(sender_name),
        sender_id=str(sender_id),
        text=text,
        timestamp=timestamp,
        reply_to_message_id=reply_to,
    )


def _build_parsed_chat(data: dict) -> ParsedChat:
    """
    Build a ParsedChat from validated raw JSON data.

    Raises:
        ValueError: if fewer than 2 participants or fewer than 50 text messages.
    """
    chat_name: str = data.get("name", "Unknown Chat")
    chat_type: str = data.get("type", "unknown")

    # Parse all eligible messages.
    messages: list[ParsedMessage] = []
    for raw_msg in data["messages"]:
        if not isinstance(raw_msg, dict):
            continue
        parsed = _parse_single_message(raw_msg)
        if parsed is not None:
            messages.append(parsed)

    # Sort chronologically.
    messages.sort(key=lambda m: m.timestamp)

    # Identify participants (exactly 2 required).
    participants = identify_participants(messages)

    # Enforce minimum message count.
    if len(messages) < 50:
        raise ValueError(
            f"Expected at least 50 text messages, found {len(messages)}. "
            "The chat export may be too short for meaningful simulation."
        )

    return ParsedChat(
        chat_name=chat_name,
        chat_type=chat_type,
        participants=participants,
        messages=messages,
    )


def parse_telegram_export(file_path: str) -> ParsedChat:
    """
    Parse a Telegram Desktop JSON export file from disk.

    Args:
        file_path: path to the ``result.json`` file.

    Returns:
        A fully populated ParsedChat.

    Raises:
        ValueError: if the file is not valid JSON or doesn't match the
            expected Telegram Desktop export format.
        FileNotFoundError: if the file does not exist.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"File is not valid JSON: {exc}") from exc
    except UnicodeDecodeError:
        # Retry with a more permissive encoding.
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"File is not valid JSON: {exc}") from exc

    validate_export(data)
    return _build_parsed_chat(data)


def parse_telegram_export_bytes(file_bytes: bytes) -> ParsedChat:
    """
    Parse a Telegram Desktop JSON export from raw bytes.

    Useful when the bot receives the export file as an upload.

    Args:
        file_bytes: raw bytes of the JSON file.

    Returns:
        A fully populated ParsedChat.

    Raises:
        ValueError: if the content is not valid JSON or doesn't match the
            expected Telegram Desktop export format.
    """
    # Try UTF-8 first, fall back to replacing bad chars.
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8", errors="replace")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Content is not valid JSON: {exc}") from exc

    validate_export(data)
    return _build_parsed_chat(data)
