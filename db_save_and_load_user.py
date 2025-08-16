import os
from dotenv import load_dotenv
import psycopg

from classes import User, Message, save_user, load_user

def main():
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL не найден в .env. Добавь его в формате postgresql://user:pass@host:port/db?sslmode=require")

    print("Подключаюсь к БД...")
    with psycopg.connect(DATABASE_URL) as conn:
        print("OK: подключение установлено.")

        # гарантируем пользователя
        user = User.ensure(conn, user_id="u_demo_001", name="Alice", cache_maxlen=5)

        # сохраним пару сообщений
        incoming = Message(author="user", content="Привет!", message_id=101, chat_id=777)
        reply    = Message(author="assistant", content="Здравствуйте! Чем могу помочь?", message_id=102, chat_id=777)
        user.persist_append_messages(conn, [incoming, reply])

        # обновим кеш последними n сообщениями
        user.refresh_last_n_from_db(conn, n=10)

        # проверим users.upsert
        user.set_additional_info("language", "ru")
        save_user(conn, user)

        # перечитаем пользователя
        loaded = load_user(conn, "u_demo_001")
        if not loaded:
            print("Пользователь не найден при чтении обратно (что-то пошло не так).")
            return

        print("\n--- Прочитанный пользователь ---")
        print("user_id:", loaded.get_user_id())
        print("name   :", loaded.get_name())
        print("additional_info:", loaded.get_additional_info())

        print("\nПоследние сообщения (JSON) из кеша после refresh:")
        for m in user.get_last_n_messages_JSON(10):
            print(m)

if __name__ == "__main__":
    main()
