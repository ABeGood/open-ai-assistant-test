import json
import psycopg
from psycopg import sql
from dotenv import load_dotenv
import os
from classes.classes import Reaction
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def create_users_table() -> None:
    """
    Creates users table if it doesn't exist.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                additional_info JSONB
            );
            """)
            conn.commit()


def create_messages_table() -> None:
    """
    Creates messages table if it doesn't exist.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                author TEXT NOT NULL,
                content TEXT NOT NULL,
                reaction JSONB,
                message_id BIGINT,
                chat_id BIGINT
            );
            """)
            
            # Create index for better performance on user message queries
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_id_id_desc
            ON messages (user_id, id DESC);
            """)
            conn.commit()


def ensure_table_exists(table_name: str) -> None:
    """
    Ensures specific table exists. Called automatically when needed.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        try:
            # Check if table exists with a simple query
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT 1 FROM {} LIMIT 1").format(
                      sql.Identifier(table_name)
                  )
                )
        except psycopg.errors.UndefinedTable:
            # Table doesn't exist, create it
            if table_name == "users":
                create_users_table()
            elif table_name == "messages":
                create_messages_table()
            else:
                raise ValueError(f"Unknown table: {table_name}")
        except Exception:
            # Create table on any other error as safety measure
            if table_name == "users":
                create_users_table()
            elif table_name == "messages":
                create_messages_table()
            else:
                raise ValueError(f"Unknown table: {table_name}")


def ensure_all_tables_exist() -> None:
    """
    Ensures all required tables exist. Called automatically when needed.
    """
    ensure_table_exists("users")
    ensure_table_exists("messages")


def save_user(user) -> None:
    """
    Upsert only to users table. Creates table if it doesn't exist.
    """
    ensure_table_exists(table_name="users")
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, name, additional_info)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET name = EXCLUDED.name,
                    additional_info = EXCLUDED.additional_info;
                """,
                (user.user_id, user.name, json.dumps(user.additional_info))
            )
            conn.commit()


def load_user(user_id: str):
    """
    Возвращает User из таблицы users (без загрузки истории).
    """
    from classes.classes import User
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, name, additional_info FROM users WHERE user_id = %s",
                (user_id,)
            )
            r = cur.fetchone()

        if not r:
            return None

        u = User(user_id=r[0], name=r[1])
        if r[2]:
            u.additional_info = r[2] if isinstance(r[2], dict) else json.loads(r[2])
        return u


def ensure_user(user_id: str, name: str, cache_maxlen: int = 200):
    """
    Ensures user exists in DB; returns User object.
    """
    from classes.classes import User
    from collections import deque
    
    # Ensure users table exists
    ensure_table_exists("users")
    
    # Try to insert user (creates if not exists, ignores if exists)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, name, additional_info)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                (user_id, name, json.dumps({}))
            )
        conn.commit()

    # Load user from DB
    loaded = load_user(user_id)
    if loaded is None:
        loaded = User(user_id=user_id, name=name, cache_maxlen=cache_maxlen)
    else:
        loaded.chat_history = deque(maxlen=cache_maxlen)
    return loaded




def get_last_n_msgs_from_db_for_user(user, n: int) -> None:
    """
    Подтягивает последние n сообщений из таблицы messages (по id DESC), складывает в кеш chronologically.
    """
    from classes.classes import Message
    from collections import deque
    
    if n <= 0:
        user.chat_history.clear()
        return

    # Ensure messages table exists
    ensure_table_exists("messages")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT author, content, reaction, message_id, chat_id
                FROM messages
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user.user_id, n)
            )
            rows = cur.fetchall()

    msgs = [
        Message(author=a, content=c, reaction=r, message_id=mid, chat_id=cid)
        for (a, c, r, mid, cid) in reversed(rows)
    ]
    user.set_chat_history(msgs)


def append_messages(user, messages: list) -> None:
    """
    Пишет сообщения в таблицу messages и добавляет их в кеш.
    """
    if not messages:
        return
    
    # Ensure messages table exists
    ensure_table_exists("messages")
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO messages (user_id, author, content, reaction, message_id, chat_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (user.user_id, m.author, m.content, m.reaction, m.message_id, m.chat_id)
                    for m in messages
                ]
            )
            conn.commit()


def update_message_reaction(user_id: str, message_id: int, reaction: Reaction) -> None:
    """
    Updates reaction for a specific message in the database.
    
    Args:
        user_id: User ID who set the reaction
        message_id: Telegram message ID
        reaction: Reaction to set (can be None to remove reaction)
    """
    ensure_table_exists("messages")
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            if reaction is None:
                # Remove reaction
                cur.execute(
                    """
                    UPDATE messages 
                    SET reaction = NULL
                    WHERE user_id = %s AND message_id = %s
                    """,
                    (user_id, message_id)
                )
            else:
                # Set reaction
                cur.execute(
                    """
                    UPDATE messages 
                    SET reaction = %s
                    WHERE user_id = %s AND message_id = %s
                    """,
                    (json.dumps({"type": reaction.name, "value": reaction.value}), user_id, message_id)
                )
            conn.commit()


def get_message_reaction(user_id: str, message_id: int) -> Reaction:
    """
    Gets current reaction for a specific message.
    
    Args:
        user_id: User ID
        message_id: Telegram message ID
        
    Returns:
        Current reaction or None if no reaction set
    """
    ensure_table_exists("messages")
    
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT reaction FROM messages 
                WHERE user_id = %s AND message_id = %s
                """,
                (user_id, message_id)
            )
            result = cur.fetchone()
            
            if not result or not result[0]:
                return None
                
            reaction_data = result[0]
            if isinstance(reaction_data, dict):
                return Reaction.from_any(reaction_data.get("type"))
            else:
                return Reaction.from_any(reaction_data)
