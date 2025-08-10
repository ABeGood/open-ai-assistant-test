import json
from collections import deque
from typing import Any, Dict, List, Optional

import psycopg 

class Message:
    def __init__(
        self,
        author: str,
        content: str,
        reaction: Optional[Any] = None  # заглушка под будущую логику реакций
    ):
        self.author: str = author
        self.content: str = content
        self.reaction: Optional[Any] = reaction

    def get_author(self) -> str:
        return self.author
    def set_author(self, author: str) -> None:
        self.author = author

    def get_content(self) -> str:
        return self.content
    def set_content(self, content: str) -> None:
        self.content = content

    def get_reaction(self) -> Optional[Any]:
        return self.reaction
    def set_reaction(self, reaction: Any) -> None:
        self.reaction = reaction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "content": self.content,
            "reaction": self.reaction
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        return cls(
            author=d["author"],
            content=d["content"],
            reaction=d.get("reaction")
        )


class User:
    def __init__(self, user_id: str, name: str, cache_maxlen: int = 200):
        self.user_id: str = user_id
        self.name: str = name
        self.chat_history: deque[Message] = deque(maxlen=cache_maxlen)
        self.additional_info: Dict[str, Any] = {}

    def get_user_id(self) -> str:
        return self.user_id
    def set_user_id(self, user_id: str) -> None:
        self.user_id = user_id

    def get_name(self) -> str:
        return self.name
    def set_name(self, name: str) -> None:
        self.name = name

    def get_chat_history(self) -> List[Message]:
        return list(self.chat_history)
    def set_chat_history(self, history: List[Message]) -> None:
        self.chat_history = deque(history, maxlen=self.chat_history.maxlen)

    def get_additional_info(self) -> Dict[str, Any]:
        return self.additional_info
    def set_additional_info(self, key: str, value: Any) -> None:
        self.additional_info[key] = value


    def add_message(self, author: str, content: str, reaction: Optional[Any] = None) -> Message:
        msg = Message(author=author, content=content, reaction=reaction)
        self.chat_history.append(msg)
        return msg

    def get_last_n_messages(self, n: int) -> List[Message]:
        if n <= 0:
            return []
        return list(self.chat_history)[-n:]

    def get_last_n_messages_JSON(self, n: int) -> List[Dict[str, Any]]:
        msgs = self.get_last_n_messages(n)
        return [m.to_dict() for m in msgs]

    # сериализация всего пользователя для записи в БД
    def to_row(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "additional_info": self.additional_info,
            "chat_history": [m.to_dict() for m in self.chat_history],
        }

# ---------- РАБОТА С БД ----------

def save_user(conn, user: User) -> None:
    """
    Простой upsert: если user_id уже есть — обновим запись.
    """
    row = user.to_row()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (user_id, name, additional_info, chat_history)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE
              SET name = EXCLUDED.name,
                  additional_info = EXCLUDED.additional_info,
                  chat_history = EXCLUDED.chat_history;
            """,
            (
                row["user_id"],
                row["name"],
                json.dumps(row["additional_info"]),      # Python -> JSON
                json.dumps(row["chat_history"]),         # список dict -> JSON
            )
        )
    conn.commit()

def load_user(conn, user_id: str) -> Optional[User]:
    """
    Пример чтения обратно из БД в объект User.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT user_id, name, additional_info, chat_history FROM users WHERE user_id = %s", (user_id,))
        r = cur.fetchone()
        if not r:
            return None

        u = User(user_id=r[0], name=r[1])
        # восстановим additional_info
        if r[2]:
            u.additional_info = r[2] if isinstance(r[2], dict) else json.loads(r[2])
        # восстановим chat_history
        if r[3]:
            ch = r[3] if isinstance(r[3], list) else json.loads(r[3])
            msgs = [Message.from_dict(d) for d in ch]
            u.set_chat_history(msgs)
        return u
