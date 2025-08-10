import os
from dotenv import load_dotenv
import psycopg

from classes import User, save_user, load_user

def main():
    # 1) Читаем строку подключения из .env
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL не найден в .env. Добавь его в формате postgresql://user:pass@host:port/db?sslmode=require")

    # 2) Создаём тестового пользователя
    user = User(user_id="u_demo_001", name="Alice", cache_maxlen=5)
    user.set_additional_info("language", "ru")
    user.set_additional_info("age", 30)
    user.add_message(author="user", content="Привет!")
    user.add_message(author="assistant", content="Здравствуйте! Чем могу помочь?")

    # 3) Подключаемся к БД и сохраняем пользователя
    print("Подключаюсь к БД...")
    with psycopg.connect(DATABASE_URL) as conn:
        print("OK: подключение установлено.")
        save_user(conn, user)
        print("OK: пользователь сохранён (upsert).")

        # 4) Загружаем этого пользователя обратно по user_id
        loaded = load_user(conn, "u_demo_001")
        if not loaded:
            print("Пользователь не найден при чтении обратно (что-то пошло не так).")
            return

        # 5) Печатаем то, что прочитали
        print("\n--- Прочитанный пользователь ---")
        print("user_id:", loaded.get_user_id())
        print("name   :", loaded.get_name())
        print("additional_info:", loaded.get_additional_info())

        print("\nПоследние сообщения (JSON):")
        for m in loaded.get_last_n_messages_JSON(10):
            print(m)

if __name__ == "__main__":
    main()
