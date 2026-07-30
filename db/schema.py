import aiosqlite
import os

DB_PATH = os.getenv("DB_PATH", "data/app.db")


async def init_db():
    """Create all tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'idle',
                group_chat_id INTEGER,
                character_a_id TEXT,
                character_b_id TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (character_a_id) REFERENCES characters(id),
                FOREIGN KEY (character_b_id) REFERENCES characters(id)
            );

            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                token TEXT NOT NULL,
                profile_json TEXT NOT NULL DEFAULT '{}',
                few_shot_examples TEXT NOT NULL DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                sender_name TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                sent_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (character_id) REFERENCES characters(id)
            );

            CREATE TABLE IF NOT EXISTS user_states (
                user_id INTEGER PRIMARY KEY,
                current_step TEXT NOT NULL DEFAULT 'idle',
                pending_data TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS original_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        await db.commit()

        # FTS5 full-text search for original messages
        await db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS original_messages_fts USING fts5(
                sender_name, text, content='original_messages', content_rowid='id'
            );

            -- Triggers to keep original_messages_fts in sync
            CREATE TRIGGER IF NOT EXISTS original_messages_ai AFTER INSERT ON original_messages BEGIN
                INSERT INTO original_messages_fts (rowid, sender_name, text)
                VALUES (new.id, new.sender_name, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS original_messages_ad AFTER DELETE ON original_messages BEGIN
                INSERT INTO original_messages_fts (original_messages_fts, rowid, sender_name, text)
                VALUES ('delete', old.id, old.sender_name, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS original_messages_au AFTER UPDATE ON original_messages BEGIN
                INSERT INTO original_messages_fts (original_messages_fts, rowid, sender_name, text)
                VALUES ('delete', old.id, old.sender_name, old.text);
                INSERT INTO original_messages_fts (rowid, sender_name, text)
                VALUES (new.id, new.sender_name, new.text);
            END;
        """)
        await db.commit()

        # FTS5 full-text search for chat history (standalone — not content-linked
        # because chat_history uses TEXT PRIMARY KEY with implicit rowid)
        await db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chat_history_fts USING fts5(
                session_id, sender_name, text
            );

            CREATE TRIGGER IF NOT EXISTS chat_history_ai AFTER INSERT ON chat_history BEGIN
                INSERT INTO chat_history_fts (rowid, session_id, sender_name, text)
                VALUES (new.rowid, new.session_id, new.sender_name, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chat_history_ad AFTER DELETE ON chat_history BEGIN
                INSERT INTO chat_history_fts (chat_history_fts, rowid, session_id, sender_name, text)
                VALUES ('delete', old.rowid, old.session_id, old.sender_name, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chat_history_au AFTER UPDATE ON chat_history BEGIN
                INSERT INTO chat_history_fts (chat_history_fts, rowid, session_id, sender_name, text)
                VALUES ('delete', old.rowid, old.session_id, old.sender_name, old.text);
                INSERT INTO chat_history_fts (rowid, session_id, sender_name, text)
                VALUES (new.rowid, new.session_id, new.sender_name, new.text);
            END;
        """)
        await db.commit()

        # Backfill FTS tables with existing data (idempotent — triggers handle future inserts)
        try:
            await db.execute(
                "INSERT OR IGNORE INTO original_messages_fts (rowid, sender_name, text) "
                "SELECT id, sender_name, text FROM original_messages"
            )
            await db.execute(
                "INSERT OR IGNORE INTO chat_history_fts (rowid, session_id, sender_name, text) "
                "SELECT rowid, session_id, sender_name, text FROM chat_history"
            )
            await db.commit()
        except Exception:
            pass  # data already indexed or tables empty

        # Migration: add memories_json column to characters table if it doesn't exist
        try:
            await db.execute("ALTER TABLE characters ADD COLUMN memories_json TEXT DEFAULT '{}'")
            await db.commit()
        except aiosqlite.OperationalError:
            pass  # column already exists

        # Migration: add style_analyzer column to characters table if it doesn't exist
        try:
            await db.execute("ALTER TABLE characters ADD COLUMN style_analyzer TEXT DEFAULT '{}'")
            await db.commit()
        except aiosqlite.OperationalError:
            pass  # column already exists
