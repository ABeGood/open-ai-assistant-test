import os
from dotenv import load_dotenv
from openai import OpenAI
import logging
import json
from time import sleep
from telebot import custom_filters
from telebot.async_telebot import AsyncTeleBot
from telebot.types import ReactionTypeEmoji
from telegram.constants import ParseMode
from telebot.apihelper import ApiTelegramException

from orchestrator_for_tg import create_orchestrator, process_telegram_message
import asyncio

DEBUG = True

load_dotenv()
telegram_token = os.environ.get("TELEGRAM_TOKEN")

logging.basicConfig(
    level=logging.DEBUG, 
    filename='bot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def format_telegram_message(response_data: dict) -> tuple[str, list[str]]:
    """
    Format response data for Telegram API with markdown-compatible formatting.
    
    Args:
        response_data (Dict[str, Any]): Dictionary containing response, sources, and images
        
    Returns:
        Tuple[str, List[str]]: (formatted_markdown_text, images_list)
    """
    
    # Extract main response text
    response_text = response_data.get('response', '')
    sources = response_data.get('sources', [])
    images = response_data.get('images', [])
    
    # Clean and format the response text for Telegram markdown
    formatted_response = escape_telegram_markdown(response_text)
    
    # Build the complete message
    message_parts = []
    
    # Add the main response
    if formatted_response:
        message_parts.append(formatted_response)
    
    # Add sources section if sources exist
    if sources:
        message_parts.append("\n\n*Источники:*")
        
        # Format each source as a markdown link on a new line
        for i, source in enumerate(sources, 1):
            if isinstance(source, str) and source.strip():
                # Extract domain name for display text
                display_text = extract_domain_name(source)
                # Format as Telegram-compatible markdown link
                link_text = f"[{display_text}]({source})"
                message_parts.append(f"{i}. {link_text}")
    
    # Join all parts
    final_message = "\n".join(message_parts)
    
    # Ensure images is a list
    images_list = images if isinstance(images, list) else []
    
    return final_message, images_list


def escape_telegram_markdown(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2 compatibility.
    
    Args:
        text (str): Original text to escape
        
    Returns:
        str: Escaped text compatible with Telegram
    """
    
    # First, convert headers (###, ##, #) to bold formatting
    text = convert_headers_to_bold(text)
    
    # Then, preserve existing markdown formatting by temporarily replacing it
    text = preserve_existing_markdown(text)
    
    # Escape special characters that could break Telegram parsing
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    # Note: Character escaping is commented out as it's not needed for current use case
    # for char in special_chars:
    #     text = text.replace(char, f'\\{char}')
    
    # Restore preserved markdown
    text = restore_preserved_markdown(text)
    
    return text


def convert_headers_to_bold(text: str) -> str:
    """Convert markdown headers (###, ##, #) to bold formatting.
    
    Args:
        text (str): Text with potential headers
    Returns:
        str: Text with headers converted to bold
    """
    import re
    
    # Store replacements to restore later
    replacements = {
        '**': '<<<BOLD>>>',
        '*': '<<<ITALIC>>>',
        '__': '<<<UNDERLINE>>>',
        '~~': '<<<STRIKE>>>',
        '`': '<<<CODE>>>',
    }
    
    # Temporarily replace existing markdown with placeholders
    for original, placeholder in replacements.items():
        text = text.replace(original, placeholder)
    
    # Pattern to match headers: 1-6 # symbols followed by space and text
    # Captures the header text without the # symbols
    header_pattern = r'^(#{1,6})\s+(.+)$'
    
    # Process each line
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check if line is a header
        match = re.match(header_pattern, line.strip())
        if match:
            header_text = match.group(2).strip()
            # Convert to bold formatting
            processed_lines.append(f"**{header_text}**")
        else:
            processed_lines.append(line)
    
    # Rejoin lines
    result = '\n'.join(processed_lines)
    
    # Restore original markdown formatting
    for original, placeholder in replacements.items():
        result = result.replace(placeholder, original)
    
    return result


def restore_preserved_markdown(text: str) -> str:
    """
    Restore preserved markdown formatting.
    """
    
    # Restore replacements
    replacements = {
        '<<<BOLD>>>': '*',
        '<<<ITALIC>>>': '_',
        '<<<UNDERLINE>>>': '__',
        '<<<STRIKE>>>': '~',
        '<<<CODE>>>': '`',
    }
    
    for placeholder, original in replacements.items():
        text = text.replace(placeholder, original)
    
    return text


def extract_domain_name(url: str) -> str:
    """
    Extract a clean domain name from URL for display.
    
    Args:
        url (str): Full URL
        
    Returns:
        str: Clean domain name
    """
    
    try:
        # Remove protocol
        if '://' in url:
            url = url.split('://', 1)[1]
        
        # Remove www prefix
        if url.startswith('www.'):
            url = url[4:]
        
        # Extract domain (before first slash)
        domain = url.split('/')[0]
        
        # Remove port if present
        domain = domain.split(':')[0]
        
        return domain
    
    except Exception:
        # Fallback to original URL if parsing fails
        return url[:50] + '...' if len(url) > 50 else url

def preserve_existing_markdown(text: str) -> str:
    """
    Temporarily replace existing markdown with placeholders.
    """
    
    # Store replacements to restore later
    replacements = {
        '**': '<<<BOLD>>>',
        '*': '<<<ITALIC>>>',
        '__': '<<<UNDERLINE>>>',
        '~~': '<<<STRIKE>>>',
        '`': '<<<CODE>>>',
    }
    
    for original, placeholder in replacements.items():
        text = text.replace(original, placeholder)
    
    return text


class TelegramBot:
    bot : AsyncTeleBot

    def __init__(self, bot_token:str) -> None:
        self.bot = AsyncTeleBot(token=bot_token)
        self.admin_messages = {}

        self.logger = logging.getLogger(__name__)

        self.register_handlers()
        self.orchestrator = create_orchestrator()

    def run_bot(self):
        """Start both the bot and reminder system"""
        
        # Start bot polling
        self.logger.info("Starting bot...")
        try:
             asyncio.run(self.bot.polling(non_stop=True, timeout=60, request_timeout=90)) # Increased timeout
        except Exception as e:
            self.logger.error(f"Bot polling stopped due to an error: {e}")
            # You might want to add a retry mechanism here, e.g.,
            sleep(5)
            self.run_bot()


    def register_handlers(self):
        """Register all message handlers"""

        @self.bot.message_handler(
            func=lambda msg: msg.text is not None and '/' not in msg.text,
        )
        async def handle_message(msg):
            if msg.text == "Hi":
                await self.bot.send_message(msg.chat.id, "Hello!", parse_mode=ParseMode.MARKDOWN)
            else:

                try:
                    await self.bot.set_message_reaction(
                        msg.chat.id,
                        msg.message_id,
                        [ReactionTypeEmoji('👍')],
                        is_big=False
                    )

                    result = await process_telegram_message(
                        self.orchestrator, 
                        msg.text, 
                        telegram_user_id=msg.from_user.id
                    )

                    debug_msg = "🔧 DEBUG MESSAGE:\n\n" \
                        f"🤖 *Routed to:* {result['specialists']}\n" \
                        # f"❓ *Reason:* {clean_message_text(result['routing_decision']['reason'])}\n" \
                        # f"📊 *Session:* {clean_message_text(result['session_id'])}" 
                    
                    await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    tg_message, images = format_telegram_message(result)

                    # Telegram caption limit
                    CAPTION_LIMIT = 1024
                    caption_too_long = len(tg_message) > CAPTION_LIMIT

                    if len(images) < 1:
                        # No images - just send text message
                        await self.bot.send_message(msg.chat.id, tg_message, parse_mode=ParseMode.MARKDOWN)
                    else:
                        if len(images) == 1:
                            # Send single image
                            with open(images[0], 'rb') as photo:
                                if caption_too_long:
                                    # Send image without caption, then text separately
                                    await self.bot.send_photo(msg.chat.id, photo)
                                    await self.bot.send_message(msg.chat.id, tg_message, parse_mode=ParseMode.MARKDOWN)
                                else:
                                    # Send image with caption
                                    await self.bot.send_photo(msg.chat.id, photo, caption=tg_message, parse_mode=ParseMode.MARKDOWN)
                        else:
                            # Send multiple images as media group
                            from telebot.types import InputMediaPhoto
                            # Keep files open during the entire operation
                            opened_files = []
                            media_group = []
                            
                            try:
                                for i, img_path in enumerate(images):
                                    photo = open(img_path, 'rb')
                                    opened_files.append(photo)
                                    
                                    # Only add caption to first image if it's not too long
                                    caption = tg_message if (i == 0 and not caption_too_long) else None
                                    media_group.append(InputMediaPhoto(photo, caption=caption, parse_mode=ParseMode.MARKDOWN if caption else None))
                                
                                await self.bot.send_media_group(msg.chat.id, media_group)
                                
                                # If caption was too long, send it as separate message after media group
                                if caption_too_long:
                                    await self.bot.send_message(msg.chat.id, tg_message, parse_mode=ParseMode.MARKDOWN)
                                    
                            finally:
                                for photo in opened_files:
                                    photo.close()

                except Exception as e:
                    # await self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}', parse_mode=ParseMode.MARKDOWN)
                    await self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}')

                    if DEBUG:
                        await self.bot.send_message(msg.chat.id, f'DEBUG MODE IS ACTIVE\n\nPlain text:\n{tg_message}')

tg_bot = TelegramBot(bot_token=str(telegram_token))
tg_bot.run_bot()
