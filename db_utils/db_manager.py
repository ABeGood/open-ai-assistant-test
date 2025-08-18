import json
import psycopg
from dotenv import load_dotenv
import os
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def create_users_table(conn) -> None:
    """
    Creates users table if it doesn't exist.
    """
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            additional_info JSONB
        );
        """)
    conn.commit()


def create_messages_table(conn) -> None:
    """
    Creates messages table if it doesn't exist.
    """
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


def ensure_table_exists(conn, table_name: str) -> None:
    """
    Ensures specific table exists. Called automatically when needed.
    """
    try:
        # Check if table exists with a simple query
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
    except psycopg.errors.UndefinedTable:
        # Table doesn't exist, create it
        if table_name == "users":
            create_users_table(conn)
        elif table_name == "messages":
            create_messages_table(conn)
        else:
            raise ValueError(f"Unknown table: {table_name}")
    except Exception:
        # Create table on any other error as safety measure
        if table_name == "users":
            create_users_table(conn)
        elif table_name == "messages":
            create_messages_table(conn)
        else:
            raise ValueError(f"Unknown table: {table_name}")


def ensure_tables_exist(conn) -> None:
    """
    Ensures all required tables exist. Called automatically when needed.
    """
    ensure_table_exists(conn, "users")
    ensure_table_exists(conn, "messages")


def save_user(conn, user) -> None:
    """
    Upsert only to users table. Creates table if it doesn't exist.
    """
    ensure_tables_exist(conn)
    
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


def load_user(conn, user_id: str):
    """
    Возвращает User из таблицы users (без загрузки истории).
    """
    from classes import User
    
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


def ensure_user(conn, user_id: str, name: str, cache_maxlen: int = 200):
    """
    Гарантирует, что пользователь есть в БД; возвращает объект User.
    """
    from classes import User
    
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

    loaded = load_user(conn, user_id)
    if loaded is None:
        loaded = User(user_id=user_id, name=name, cache_maxlen=cache_maxlen)
    else:
        from collections import deque
        loaded.chat_history = deque(maxlen=cache_maxlen)
    return loaded


def refresh_user_last_n_from_db(conn, user, n: int) -> None:
    """
    Подтягивает последние n сообщений из таблицы messages (по id DESC), складывает в кеш chronologically.
    """
    from classes import Message
    from collections import deque
    
    if n <= 0:
        user.chat_history.clear()
        return

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


def persist_append_messages(conn, user, messages: list) -> None:
    """
    Пишет сообщения в таблицу messages и добавляет их в кеш.
    """
    if not messages:
        return
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
    for m in messages:
        user.chat_history.append(m)