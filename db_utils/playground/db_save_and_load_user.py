import os
import sys
from dotenv import load_dotenv
import psycopg

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from classes.classes import User, Message
from db_utils.db_manager import save_user, load_user, ensure_user

def drop_and_recreate_tables():
    """Drop existing tables and recreate them with new schema"""
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL не найден в .env. Добавь его в формате postgresql://user:pass@host:port/db?sslmode=require")

    print("Подключаюсь к БД...")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            print("Удаляем старые таблицы...")
            cur.execute("DROP TABLE IF EXISTS messages CASCADE;")
            cur.execute("DROP TABLE IF EXISTS users CASCADE;")
            
            print("Создаем новые таблицы...")
            # Create users table with last_active field
            cur.execute("""
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    additional_info JSONB,
                    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create messages table with created_at, images and reaction fields
            cur.execute("""
                CREATE TABLE messages (
                    id SERIAL PRIMARY KEY,
                    author VARCHAR(255) NOT NULL,
                    content TEXT,
                    reaction INTEGER DEFAULT NULL,
                    message_id INTEGER,
                    chat_id INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    images JSONB DEFAULT '[]'::jsonb
                );
            """)
            
            # Add indexes for better performance
            cur.execute("CREATE INDEX idx_users_user_id ON users(user_id);")
            cur.execute("CREATE INDEX idx_messages_author ON messages(author);")
            cur.execute("CREATE INDEX idx_messages_created_at ON messages(created_at);")
            
            conn.commit()
            print("Таблицы созданы успешно!")

def main():
    # First drop and recreate tables
    drop_and_recreate_tables()
    
    print("\nТестируем новую схему...")

    # # Test the new schema
    # print("Создаем тестового пользователя...")
    # user = ensure_user(user_id="u_demo_001", name="Alice", cache_maxlen=5)

    # # Create messages with new fields
    # incoming = Message(
    #     author="u_demo_001", 
    #     content="Привет!", 
    #     message_id=101, 
    #     chat_id=777,
    #     images=["image1.png", "image2.jpg"]
    # )
    # reply = Message(
    #     author="bot_to_u_demo_001", 
    #     content="Здравствуйте! Чем могу помочь?", 
    #     message_id=102, 
    #     chat_id=777,
    #     images=[]
    # )
    
    # print("Сохраняем сообщения...")
    # user.persist_append_messages([incoming, reply])

    # # Update user info and save
    # user.set_additional_info("language", "ru")
    # user.save()

    # print("Перечитываем пользователя из БД...")
    # loaded = load_user("u_demo_001")
    # if not loaded:
    #     print("Пользователь не найден при чтении обратно (что-то пошло не так).")
    #     return

    # print("\n--- Прочитанный пользователь ---")
    # print("user_id:", loaded.get_user_id())
    # print("name   :", loaded.get_name())
    # print("last_active:", loaded.get_last_active())
    # print("additional_info:", loaded.get_additional_info())

    # print("\nПоследние сообщения из кеша:")
    # chat_history = loaded.get_chat_history(10)
    # for msg in chat_history:
    #     print(f"Author: {msg.get_author()}")
    #     print(f"Content: {msg.get_content()}")
    #     print(f"Created: {msg.get_created_at()}")
    #     print(f"Images: {msg.get_images()}")
    #     print("---")

if __name__ == "__main__":
    main()
