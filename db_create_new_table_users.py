# скрипт для создания таблицы users в PostgreSQL
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
        # создаём таблицу users (lowercase), JSONB для гибких полей
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            additional_info JSONB,
            chat_history JSONB
        );
        """)
        conn.commit()
        print("OK: таблица users создана/проверена.")
        
