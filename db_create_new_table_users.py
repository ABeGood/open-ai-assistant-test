# скрипт для создания таблиц users и messages в PostgreSQL
import os
from dotenv import load_dotenv
import psycopg

load_dotenv()  # читает .env рядом со скриптом

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL не найден в .env")

print("Пробую подключиться к БД...")
with psycopg.connect(DATABASE_URL) as conn:
    print("OK: подключение есть.")
    with conn.cursor() as cur:
        # 1) таблица users: БЕЗ chat_history
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            additional_info JSONB
        );
        """)

        # 2) таблица messages: история сообщений
        # добавляю surrogate PK id для надёжной сортировки и ссылочную целостность по user_id
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

        # ускорим выбор последних сообщений пользователя
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_user_id_id_desc
        ON messages (user_id, id DESC);
        """)

        conn.commit()
        print("OK: таблицы users и messages созданы/проверены.")
