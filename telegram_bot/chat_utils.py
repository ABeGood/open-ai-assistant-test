from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from classes.classes import Reaction
from db_utils.db_manager import update_message_reaction, get_message_reaction


def create_reaction_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """
    Creates inline keyboard with like/dislike buttons for message reactions.
    
    Args:
        message_id: ID of the bot message to attach reactions to
        
    Returns:
        InlineKeyboardMarkup with reaction buttons
    """
    keyboard = InlineKeyboardMarkup()
    
    like_button = InlineKeyboardButton(
        text="👍", 
        callback_data=f"like:{message_id}"
    )
    dislike_button = InlineKeyboardButton(
        text="👎", 
        callback_data=f"dislike:{message_id}"
    )
    
    keyboard.add(like_button, dislike_button)
    return keyboard


def create_locked_reaction_keyboard(message_id: int) -> InlineKeyboardMarkup:
    """
    Creates locked (disabled) keyboard with transparent like/dislike buttons.
    
    Args:
        message_id: ID of the bot message to attach reactions to
        
    Returns:
        InlineKeyboardMarkup with locked reaction buttons
    """
    keyboard = InlineKeyboardMarkup()
    
    # Use transparent emoji and same callback_data to maintain structure
    like_button = InlineKeyboardButton(
        text="⚪",  # Transparent/disabled indicator
        callback_data=f"locked:{message_id}"
    )
    dislike_button = InlineKeyboardButton(
        text="⚪",  # Transparent/disabled indicator
        callback_data=f"locked:{message_id}"
    )
    
    keyboard.add(like_button, dislike_button)
    return keyboard


def parse_reaction_callback(callback_data: str) -> tuple[str, int]:
    """
    Parses reaction callback data to extract reaction type and message ID.
    
    Args:
        callback_data: Callback data in format "like:123", "dislike:123", or "locked:123"
        
    Returns:
        Tuple of (reaction_type, message_id)
        
    Raises:
        ValueError: If callback_data format is invalid
    """
    try:
        reaction_type, message_id_str = callback_data.split(":")
        message_id = int(message_id_str)
        return reaction_type, message_id
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid callback data format: {callback_data}")


def get_reaction_emoji(reaction: Reaction) -> str:
    """
    Returns emoji for the given reaction.
    
    Args:
        reaction: Reaction enum value
        
    Returns:
        Emoji string for the reaction
    """
    if reaction == Reaction.LIKE:
        return "👍"
    elif reaction == Reaction.DISLIKE:
        return "👎"
    else:
        return ""
