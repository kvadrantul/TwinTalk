import re
import logging

logger = logging.getLogger(__name__)

# Telegram bot token format: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TOKEN_PATTERN = re.compile(r'^\d{8,10}:[A-Za-z0-9_-]{35}$')


def validate_token_format(token: str) -> bool:
    """
    Quick format check for Telegram bot token.
    Does NOT validate against Telegram API (use validate_bot_token for that).
    """
    return bool(TOKEN_PATTERN.match(token.strip()))


def validate_json_file(file_bytes: bytes) -> tuple[bool, str]:
    """
    Quick validation of uploaded JSON file before full parsing.
    Returns (is_valid, error_message).
    """
    import json
    
    # Check file size (max 50MB)
    if len(file_bytes) > 50 * 1024 * 1024:
        return False, "Файл слишком большой (максимум 50 МБ)."
    
    # Check it's valid JSON
    try:
        data = json.loads(file_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, "Файл не является корректным JSON."
    
    # Check it looks like a Telegram export
    if not isinstance(data, dict):
        return False, "Неверный формат файла."
    
    if "messages" not in data:
        return False, "Файл не содержит ключ 'messages'. Это не похоже на экспорт Telegram Desktop."
    
    if not isinstance(data["messages"], list):
        return False, "Поле 'messages' должно быть массивом."
    
    if len(data["messages"]) == 0:
        return False, "Файл не содержит сообщений."
    
    return True, ""


def validate_speed_multiplier(value: str) -> tuple[bool, float]:
    """
    Validate speed multiplier argument.
    Returns (is_valid, multiplier_value).
    """
    try:
        multiplier = float(value)
        if multiplier in (0.5, 1.0, 2.0, 4.0):
            return True, multiplier
        return False, 0
    except ValueError:
        return False, 0


def sanitize_text(text: str, max_length: int = 4096) -> str:
    """
    Sanitize text before sending to Telegram.
    - Trim to max_length (Telegram message limit is 4096)
    - Remove null bytes
    """
    text = text.replace('\x00', '')
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."
    return text
