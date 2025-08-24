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
                additional_info JSONB,
                last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW()
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
                author TEXT NOT NULL,
                content TEXT NOT NULL,
                reaction INTEGER,
                message_id BIGINT,
                chat_id BIGINT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                images JSONB DEFAULT '[]'::jsonb
            );
            """)
            
            # Create index for better performance on author queries
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_author
            ON messages (author);
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
                INSERT INTO users (user_id, name, additional_info, last_active)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET name = EXCLUDED.name,
                    additional_info = EXCLUDED.additional_info,
                    last_active = EXCLUDED.last_active;
                """,
                (user.user_id, user.name, json.dumps(user.additional_info), user.last_active)
            )
            conn.commit()


def load_user(user_id: str):
    """
    Возвращает User из таблицы users и загружает историю сообщений.
    """
    from classes.classes import User, Message
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, name, additional_info, last_active FROM users WHERE user_id = %s",
                (user_id,)
            )
            r = cur.fetchone()

        if not r:
            return None

        u = User(user_id=r[0], name=r[1], last_active=r[3])
        if r[2]:
            u.additional_info = r[2] if isinstance(r[2], dict) else json.loads(r[2])
        
        # Load chat history including bot responses
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT author, content, reaction, message_id, chat_id, created_at, images 
                FROM messages 
                WHERE author = %s OR author LIKE %s 
                ORDER BY created_at ASC
                """,
                (user_id, f"bot_to_%")
            )
            messages = cur.fetchall()
            
            for msg_data in messages:
                images = []
                if msg_data[6]:  # images field (now at index 6)
                    if isinstance(msg_data[6], str):
                        try:
                            images = json.loads(msg_data[6])
                        except json.JSONDecodeError:
                            images = []
                    elif isinstance(msg_data[6], list):
                        images = msg_data[6]
                
                # Parse reaction
                reaction = None
                if msg_data[2] is not None:  # reaction field
                    reaction = Reaction.from_any(msg_data[2])
                
                msg = Message(
                    author=msg_data[0],
                    content=msg_data[1],
                    reaction=reaction,
                    message_id=msg_data[3],
                    chat_id=msg_data[4],
                    created_at=msg_data[5],
                    images=images
                )
                u.chat_history.append(msg)
        
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
            from datetime import datetime
            cur.execute(
                """
                INSERT INTO users (user_id, name, additional_info, last_active)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                (user_id, name, json.dumps({}), datetime.now())
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
                SELECT author, content, reaction, message_id, chat_id, created_at, images
                FROM messages
                WHERE author = %s OR author LIKE %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user.user_id, f"bot_to_%", n)
            )
            rows = cur.fetchall()

    msgs = []
    for (author, content, reaction, message_id, chat_id, created_at, images) in reversed(rows):
        # Parse images
        parsed_images = []
        if images:
            if isinstance(images, str):
                try:
                    parsed_images = json.loads(images)
                except json.JSONDecodeError:
                    parsed_images = []
            elif isinstance(images, list):
                parsed_images = images
        
        # Parse reaction
        parsed_reaction = None
        if reaction is not None:
            parsed_reaction = Reaction.from_any(reaction)
            
        msg = Message(
            author=author, 
            content=content, 
            reaction=parsed_reaction, 
            message_id=message_id, 
            chat_id=chat_id,
            created_at=created_at,
            images=parsed_images
        )
        msgs.append(msg)
    
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
                INSERT INTO messages (author, content, reaction, message_id, chat_id, created_at, images)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        m.author, 
                        m.content, 
                        m.reaction.value if m.reaction else None, 
                        m.message_id, 
                        m.chat_id, 
                        m.created_at, 
                        json.dumps(m.images)
                    )
                    for m in messages
                ]
            )
            conn.commit()


def update_message_reaction(user_id: str, message_id: int, reaction: Reaction|None) -> None:
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
                # Remove reaction - update bot message with this message_id
                cur.execute(
                    """
                    UPDATE messages 
                    SET reaction = NULL
                    WHERE author = %s AND message_id = %s
                    """,
                    (f"bot_to_{message_id}", message_id)
                )
            else:
                # Set reaction - update bot message with this message_id
                cur.execute(
                    """
                    UPDATE messages 
                    SET reaction = %s
                    WHERE author = %s AND message_id = %s
                    """,
                    (reaction.value, f"bot_to_{message_id}", message_id)
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
                WHERE author = %s AND message_id = %s
                """,
                (f"bot_to_{message_id}", message_id)
            )
            result = cur.fetchone()
            
            if not result or result[0] is None:
                return None
                
            reaction_data = result[0]
            return Reaction.from_any(reaction_data)
