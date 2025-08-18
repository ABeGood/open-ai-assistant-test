from collections import deque
from typing import Any, Dict, List, Optional


class Message:
    def __init__(
        self,
        author: str,
        content: str,
        reaction: Optional[Any] = None,  # заглушка под будущую логику реакций
        message_id: Optional[int] = None,
        chat_id: Optional[int] = None,
    ):
        self.author: str = author
        self.content: str = content
        self.reaction: Optional[Any] = reaction
        self.message_id: Optional[int] = message_id
        self.chat_id: Optional[int] = chat_id

    # --- getters / setters ---
    def get_author(self) -> str: return self.author
    def set_author(self, author: str) -> None: self.author = author

    def get_content(self) -> str: return self.content
    def set_content(self, content: str) -> None: self.content = content

    def get_reaction(self) -> Optional[Any]: return self.reaction
    def set_reaction(self, reaction: Any) -> None: self.reaction = reaction

    def get_message_id(self) -> Optional[int]: return self.message_id
    def set_message_id(self, message_id: Optional[int]) -> None: self.message_id = message_id

    def get_chat_id(self) -> Optional[int]: return self.chat_id
    def set_chat_id(self, chat_id: Optional[int]) -> None: self.chat_id = chat_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "content": self.content,
            "reaction": self.reaction,
            "message_id": self.message_id,
            "chat_id": self.chat_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        return cls(
            author=d["author"],
            content=d["content"],
            reaction=d.get("reaction"),
            message_id=d.get("message_id"),
            chat_id=d.get("chat_id"),
        )


class User:
    def __init__(self, user_id: str, name: str, cache_maxlen: int = 200):
        self.user_id: str = user_id
        self.name: str = name
        self.chat_history: deque[Message] = deque(maxlen=cache_maxlen) # кеш последних n сообщений. существует только внутри класса User
        self.additional_info: Dict[str, Any] = {}

    # --- getters / setters ---
    def get_user_id(self) -> str: return self.user_id
    def set_user_id(self, user_id: str) -> None: self.user_id = user_id

    def get_name(self) -> str: return self.name
    def set_name(self, name: str) -> None: self.name = name

    def get_chat_history(self) -> List[Message]: return list(self.chat_history)
    def set_chat_history(self, history: List[Message]) -> None:
        self.chat_history = deque(history, maxlen=self.chat_history.maxlen)

    def get_additional_info(self) -> Dict[str, Any]: return self.additional_info
    def set_additional_info(self, key: str, value: Any) -> None:
        self.additional_info[key] = value

    # --- cache operations ---
    def add_message(self, author: str, content: str, reaction: Optional[Any] = None,
                    message_id: Optional[int] = None, chat_id: Optional[int] = None) -> Message:
        msg = Message(author=author, content=content, reaction=reaction,
                      message_id=message_id, chat_id=chat_id)
        self.chat_history.append(msg)
        return msg

    def get_last_n_messages(self, n: int) -> List[Message]:
        if n <= 0:
            return []
        return list(self.chat_history)[-n:]

    def get_last_n_messages_JSON(self, n: int) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.get_last_n_messages(n)]

    def ensure(self, conn, user_id: str, name: str, cache_maxlen: int = 200) -> "User":
        """
        Ensures user exists in DB; returns User object.
        """
        from db_utils.db_manager import ensure_user
        return ensure_user(conn, user_id, name, cache_maxlen)

    def save(self, conn) -> None:
        """
        Saves this user to database.
        """
        from db_utils.db_manager import save_user
        save_user(conn, self)

    def refresh_last_n_from_db(self, conn, n: int) -> None:
        """
        Loads last n messages from messages table (by id DESC), puts them in cache chronologically.
        """
        from db_utils.db_manager import refresh_user_last_n_from_db
        refresh_user_last_n_from_db(conn, self, n)

    def persist_append_messages(self, conn, messages: List[Message]) -> None:
        """
        Writes messages to messages table and adds them to cache.
        """
        from db_utils.db_manager import persist_append_messages
        persist_append_messages(conn, self, messages)

