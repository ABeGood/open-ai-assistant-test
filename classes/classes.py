from collections import deque
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import json

class Reaction(Enum):
    LIKE = 1
    DISLIKE = -1
    UNKNOWN = 0

    @classmethod
    def from_any(cls, raw: Any) -> Optional["Reaction"]:
        """
        Unified parser: accepts None, an Enum name ('LIKE'), or a number (1/-1/0).
        Returns None (if raw is None) or a specific Reaction.
        Unknown values → Reaction.UNKNOWN.
        """
        if raw is None:
            return None
        if isinstance(raw, Reaction):
            return raw
        if isinstance(raw, str):
            # try by name ('LIKE'/'DISLIKE'/'UNKNOWN')
            try:
                return cls[raw]
            except KeyError:
                return cls.UNKNOWN
        # try as number
        try:
            return cls(int(raw))
        except Exception:
            return cls.UNKNOWN

class Message:
    def __init__(
        self,
        author: str,
        content: str,
        reaction: Optional[Reaction] = None,  # заглушка под будущую логику реакций
        message_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        images: Optional[List[str]] = None,
    ):
        self.author: str = author
        self.content: str = content
        self.reaction: Optional[Reaction] = reaction
        self.message_id: Optional[int] = message_id
        self.chat_id: Optional[int] = chat_id
        self.created_at: datetime = created_at or datetime.now()
        self.images: List[str] = images or []

    # --- getters / setters ---
    def get_author(self) -> str: return self.author
    # def set_author(self, author: str) -> None: self.author = author

    def get_content(self) -> str: return self.content
    def set_content(self, content: str) -> None: self.content = content

    def get_reaction(self) -> Optional[Reaction]: return self.reaction
    def set_reaction(self, reaction: Optional[Reaction]) -> None: self.reaction = reaction

    def get_message_id(self) -> Optional[int]: return self.message_id
    # def set_message_id(self, message_id: Optional[int]) -> None: self.message_id = message_id

    def get_chat_id(self) -> Optional[int]: return self.chat_id
    # def set_chat_id(self, chat_id: Optional[int]) -> None: self.chat_id = chat_id

    def get_created_at(self) -> datetime: return self.created_at
    def set_created_at(self, created_at: datetime) -> None: self.created_at = created_at

    def get_images(self) -> List[str]: return self.images
    def set_images(self, images: List[str]) -> None: self.images = images

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "content": self.content,
            "reaction": self.reaction.name if self.reaction is not None else None,
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "images": self.images,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        created_at = None
        if d.get("created_at"):
            if isinstance(d["created_at"], str):
                created_at = datetime.fromisoformat(d["created_at"])
            elif isinstance(d["created_at"], datetime):
                created_at = d["created_at"]
        
        return cls(
            author=d["author"],
            content=d["content"],
            reaction=Reaction.from_any(d.get("reaction")),
            message_id=d.get("message_id"),
            chat_id=d.get("chat_id"),
            created_at=created_at,
            images=d.get("images", []),
        )


class User:
    def __init__(self, user_id: str, name: str, cache_maxlen: int = 200, last_active: Optional[datetime] = None):
        self.user_id: str = user_id
        self.name: str = name
        self.chat_history: deque[Message] = deque(maxlen=cache_maxlen) # кеш последних n сообщений. существует только внутри класса User
        self.additional_info: Dict[str, Any] = {}
        self.last_active: datetime = last_active or datetime.now()

    # --- getters / setters ---
    def get_user_id(self) -> str: return self.user_id
    # def set_user_id(self, user_id: str) -> None: self.user_id = user_id

    def get_name(self) -> str: return self.name
    # def set_name(self, name: str) -> None: self.name = name

    def get_last_active(self) -> datetime: return self.last_active
    def set_last_active(self, last_active: datetime) -> None: self.last_active = last_active

    def get_chat_history(self, last_n: int = None) -> List[Message]:
        """
        Get chat history from the loaded messages.
        
        Args:
            last_n: Number of recent messages to return. If None, returns all messages.
            
        Returns:
            List of Message objects
        """
        if last_n is None:
            return list(self.chat_history)
        return list(self.chat_history)[-last_n:] if last_n > 0 else []
    def set_chat_history(self, history: List[Message]) -> None:
        self.chat_history = deque(history, maxlen=self.chat_history.maxlen)

    def get_additional_info(self) -> Dict[str, Any]: return self.additional_info
    def set_additional_info(self, key: str, value: Any) -> None:
        self.additional_info[key] = value

    # --- cache operations ---
    def add_message(self, msg:Message) -> Message:
        self.chat_history.append(msg)
        return msg

    def get_last_n_messages(self, n: int) -> List[Message]:
        if n <= 0:
            return []
        return list(self.chat_history)[-n:]

    def get_last_n_messages_JSON(self, n: int) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.get_last_n_messages(n)]

    @staticmethod
    def ensure(user_id: str, name: str, cache_maxlen: int = 200) -> "User":
        """
        Ensures user exists in DB; returns User object.
        """
        from db_utils.db_manager import ensure_user
        return ensure_user(user_id, name, cache_maxlen)

    def save(self) -> None:
        """
        Saves this user to database.
        """
        from db_utils.db_manager import save_user
        save_user(self)

    def get_last_n_msgs_from_db(self, n: int) -> None:
        """
        Loads last n messages from messages table (by id DESC), puts them in cache chronologically.
        """
        from db_utils.db_manager import get_last_n_msgs_from_db_for_user
        get_last_n_msgs_from_db_for_user(self, n)

    def persist_append_messages(self, messages: List[Message]) -> None:
        """
        Writes messages to messages table and adds them to cache.
        """
        from db_utils.db_manager import append_messages
        append_messages(self, messages)
        for msg in messages:
            self.chat_history.append(msg)

