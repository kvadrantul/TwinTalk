"""
Parser module for Telegram Desktop JSON exports.
"""

from .export_parser import (
    ParsedMessage,
    ParsedChat,
    parse_telegram_export,
    parse_telegram_export_bytes,
    extract_text,
    identify_participants,
    validate_export,
)
from .profiler import (
    CharacterProfile,
    build_character_profile,
    select_few_shot_examples,
    profile_to_dict,
    dict_to_profile,
)

__all__ = [
    "ParsedMessage",
    "ParsedChat",
    "parse_telegram_export",
    "parse_telegram_export_bytes",
    "extract_text",
    "identify_participants",
    "validate_export",
    "CharacterProfile",
    "build_character_profile",
    "select_few_shot_examples",
    "profile_to_dict",
    "dict_to_profile",
]
