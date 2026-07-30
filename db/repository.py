import json
import aiosqlite
from typing import Optional, Union

from db.schema import DB_PATH


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict:
    """Convert a sqlite row to a dict using cursor description."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


async def _execute(query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False):
    """Helper to execute a query and optionally fetch results."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            if fetch_one:
                row = await cursor.fetchone()
                return dict(row) if row else None
            if fetch_all:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
            await db.commit()
            return None


# ── Sessions ──────────────────────────────────────────────────────────────────

async def create_session(session_id: str, user_id: int) -> dict:
    await _execute(
        "INSERT INTO sessions (id, user_id, status) VALUES (?, ?, 'idle')",
        (session_id, user_id),
    )
    return await get_session(session_id)  # type: ignore[return-value]


async def get_session(session_id: str) -> Optional[dict]:
    return await _execute("SELECT * FROM sessions WHERE id = ?", (session_id,), fetch_one=True)


async def update_session_status(session_id: str, status: str) -> None:
    await _execute("UPDATE sessions SET status = ? WHERE id = ?", (status, session_id))


async def update_session_group(session_id: str, group_chat_id: int) -> None:
    await _execute("UPDATE sessions SET group_chat_id = ? WHERE id = ?", (group_chat_id, session_id))


async def update_session_characters(session_id: str, char_a_id: str, char_b_id: str) -> None:
    await _execute(
        "UPDATE sessions SET character_a_id = ?, character_b_id = ? WHERE id = ?",
        (char_a_id, char_b_id, session_id),
    )


async def get_sessions_by_user(user_id: int) -> list[dict]:
    return await _execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
        fetch_all=True,
    )


# ── Characters ────────────────────────────────────────────────────────────────

async def create_character(
    char_id: str,
    session_id: str,
    name: str,
    token: str,
    profile_json: Union[dict, str] = "{}",
    few_shot_examples: Union[list, str] = "[]",
) -> dict:
    profile_str = json.dumps(profile_json) if isinstance(profile_json, dict) else profile_json
    examples_str = json.dumps(few_shot_examples) if isinstance(few_shot_examples, list) else few_shot_examples
    await _execute(
        "INSERT INTO characters (id, session_id, name, token, profile_json, few_shot_examples) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (char_id, session_id, name, token, profile_str, examples_str),
    )
    return await get_character(char_id)  # type: ignore[return-value]


async def get_character(char_id: str) -> Optional[dict]:
    row = await _execute("SELECT * FROM characters WHERE id = ?", (char_id,), fetch_one=True)
    if row:
        row["profile_json"] = json.loads(row["profile_json"])
        row["few_shot_examples"] = json.loads(row["few_shot_examples"])
        row["memories_json"] = json.loads(row.get("memories_json") or "{}")
        row["style_analyzer"] = json.loads(row.get("style_analyzer") or "{}")
    return row


async def get_characters_by_session(session_id: str) -> list[dict]:
    rows = await _execute(
        "SELECT * FROM characters WHERE session_id = ?", (session_id,), fetch_all=True
    )
    for row in rows:
        row["profile_json"] = json.loads(row["profile_json"])
        row["few_shot_examples"] = json.loads(row["few_shot_examples"])
        row["memories_json"] = json.loads(row.get("memories_json") or "{}")
        row["style_analyzer"] = json.loads(row.get("style_analyzer") or "{}")
    return rows


async def update_character_memories(char_id: str, memories_json: dict) -> None:
    await _execute(
        "UPDATE characters SET memories_json = ? WHERE id = ?",
        (json.dumps(memories_json), char_id),
    )


async def update_character_style_analyzer(char_id: str, style_analyzer: dict) -> None:
    await _execute(
        "UPDATE characters SET style_analyzer = ? WHERE id = ?",
        (json.dumps(style_analyzer), char_id),
    )


# ── Chat History ──────────────────────────────────────────────────────────────

async def add_message(msg_id: str, session_id: str, character_id: str, sender_name: str, text: str, turn_number: int) -> None:
    await _execute(
        "INSERT INTO chat_history (id, session_id, character_id, sender_name, text, turn_number) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, session_id, character_id, sender_name, text, turn_number),
    )


async def get_messages_by_session(session_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    return await _execute(
        "SELECT * FROM chat_history WHERE session_id = ? ORDER BY turn_number ASC LIMIT ? OFFSET ?",
        (session_id, limit, offset),
        fetch_all=True,
    )


async def get_last_messages(session_id: str, count: int = 15) -> list[dict]:
    # Fetch last N by turn_number DESC, then reverse to chronological order
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM chat_history WHERE session_id = ? ORDER BY turn_number DESC LIMIT ?",
            (session_id, count),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in reversed(rows)]


async def get_message_count(session_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM chat_history WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


# ── User States ───────────────────────────────────────────────────────────────

async def set_user_state(user_id: int, current_step: str, pending_data: Union[dict, str] = "{}") -> None:
    data_str = json.dumps(pending_data) if isinstance(pending_data, dict) else pending_data
    await _execute(
        "INSERT INTO user_states (user_id, current_step, pending_data) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET current_step = excluded.current_step, pending_data = excluded.pending_data",
        (user_id, current_step, data_str),
    )


async def get_user_state(user_id: int) -> Optional[dict]:
    row = await _execute("SELECT * FROM user_states WHERE user_id = ?", (user_id,), fetch_one=True)
    if row:
        row["pending_data"] = json.loads(row["pending_data"])
    return row


async def clear_user_state(user_id: int) -> None:
    await _execute("DELETE FROM user_states WHERE user_id = ?", (user_id,))


# ── Original Messages ────────────────────────────────────────────────────────

async def save_original_messages(session_id: str, messages: list[dict]) -> None:
    """Save original chat messages. Each dict has: sender_name, text, timestamp"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO original_messages (session_id, sender_name, text, timestamp) VALUES (?, ?, ?, ?)",
            [(session_id, m["sender_name"], m["text"], m["timestamp"]) for m in messages],
        )
        await db.commit()


async def get_original_messages(session_id: str) -> list[dict]:
    """Get all original messages for a session, ordered by timestamp."""
    return await _execute(
        "SELECT sender_name, text, timestamp FROM original_messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,),
        fetch_all=True,
    )


async def delete_original_messages(session_id: str) -> None:
    """Delete original messages for a session."""
    await _execute(
        "DELETE FROM original_messages WHERE session_id = ?",
        (session_id,),
    )


async def search_original_messages(session_id: str, query: str, limit: int = 30) -> list[dict]:
    """Full-text search across ALL original messages for this session."""
    return await _execute(
        """SELECT om.sender_name, om.text, om.timestamp
           FROM original_messages_fts fts
           JOIN original_messages om ON om.id = fts.rowid
           WHERE original_messages_fts MATCH ? AND om.session_id = ?
           ORDER BY rank
           LIMIT ?""",
        (query, session_id, limit),
        fetch_all=True,
    )
