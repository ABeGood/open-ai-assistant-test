import os
from dotenv import load_dotenv
import re
from openai import OpenAI
import logging
import telebot # telebot
import json
from time import sleep
from telebot import custom_filters, TeleBot
from telebot.types import ReactionTypeEmoji
from telegram.constants import ParseMode
from telebot.apihelper import ApiTelegramException
import re
from orchestrator_for_tg import create_orchestrator, process_telegram_message
import glob

DEBUG = True

load_dotenv()
telegram_token = os.environ.get("TELEGRAM_TOKEN")


IMG_PATTERN = r"D_\d+_IMG_\d+"

def get_all_markers_as_list(text: str) -> list[str]:
    """
    Finds all occurrences of the @_IMG_NNN pattern in a text
    and returns them as a list of strings.

    Args:
        text: The input string to search.

    Returns:
        A list of strings, where each string is a detected marker.
        Returns an empty list if no markers are found.
    """
    matches = re.findall(IMG_PATTERN, text)
    return matches

def remove_all_markers(text: str) -> str:
    """
    Removes all occurrences of the @D_N_IMG_NNN pattern from a text.

    Args:
        text: The input string to clean.

    Returns:
        A string with all markers removed.
        Returns the original string if no markers are found.
    """
    cleaned_text = re.sub(IMG_PATTERN, "", text)
    return cleaned_text

def extract_marker_parts(marker: str) -> tuple[str, str] | None:
    """
    Extracts the two main components from a marker string.
    
    Args:
        marker: A marker string like "@D_3_IMG_002"
        
    Returns:
        A tuple containing (prefix_part, img_part), e.g., ("D_3", "IMG_002")
        Returns None if the marker doesn't match the expected format.
    """
    # Pattern with two capturing groups
    pattern = r"(D_\d+)_(IMG_\d+)"
    
    match = re.match(pattern, marker)
    if match:
        return match.group(1), match.group(2)
    return None

def clean_message_text(text):
    """Clean text by removing problematic characters"""
    # Remove backslashes
    text = text.replace("\\", "")
    
    # Remove other potentially problematic characters
    problematic_chars = ['*', '_', '[', ']', '~', '`', '>', '#', '=', '|', '{', '}']
    for char in problematic_chars:
        text = text.replace(char, "")
    
    # Clean up whitespace
    text = " ".join(text.split())
    
    return text

def delete_sources_from_text(text: str):
    pattern = r'【.*?】'
    return re.sub(pattern, '', text)

def find_file_by_name(folder_path: str, filename_base: str) -> list[str]:
    """
    Finds all files with the given base name regardless of extension.
    
    Args:
        folder_path: Path to the folder to search in
        filename_base: Base filename without extension (e.g., "IMG_001")
        
    Returns:
        List of full file paths that match the base name
    """
    # Create search pattern
    search_pattern = os.path.join(folder_path, f"{filename_base}.*")
    
    # Find all matching files
    matching_files = glob.glob(search_pattern)
    
    return matching_files

def format_telegram_message(response_data):
    """
    Convert LLM agent response to Telegram message format
    
    Args:
        response_data (dict or str): LLM agent response dictionary or JSON string
    
    Returns:
        str: Formatted Telegram message
    """
    # Parse JSON string if needed
    if isinstance(response_data, str):
        response_data = json.loads(response_data)

    # CATEGORY START

    files_path = None
    assistant_used = response_data['assistant_used']
    if assistant_used == 'equipment':
        files_path = 'files_preproc/equipment/'
    elif assistant_used == 'cables':
        files_path = 'files_preproc/cables/'
    elif assistant_used == 'tools':
        files_path = 'files_preproc/tools/'
    elif assistant_used == 'commonInfo':
        files_path = 'files_preproc/common_info/'

    # CATEGORY END
    
    # Extract main response text
    # main_response = clean_message_text(response_data['response']['response'])
    main_response = delete_sources_from_text(response_data['response']['response'])
    

    # SOURCES START
    # Extract and format sources
    if files_path:
        sources = response_data['response']['sources']
        formatted_sources = []
    
        for source in sources:
            # Get filename without extension
            filename = source.filename
            name_without_ext = os.path.splitext(filename)[0]
            
            try:
                source_mapping_filepath = files_path + 'pdf_mapping.json'
                with open(source_mapping_filepath, 'r', encoding='utf-8') as file:
                    pdf_mapping = json.load(file)
                link = pdf_mapping[name_without_ext]
                # Create markdown link
                markdown_link = f"[{name_without_ext.split('_')[0]}]({link})"
                formatted_sources.append(markdown_link)
                formatted_sources = list(set(formatted_sources))
            except Exception as e:
                markdown_link = f"{filename}"
                formatted_sources.append(markdown_link)
                formatted_sources = list(set(formatted_sources))

    # SOURCES END


    # IMAGES START
    img_markers = get_all_markers_as_list(main_response)
    if img_markers and files_path:
        img_mapping_filepath = files_path + 'doc_mapping.json'
        with open(img_mapping_filepath, 'r', encoding='utf-8') as file:
            img_mapping = json.load(file)
        
        img_list = []
        for img in img_markers:
            img_file_key, img_name = extract_marker_parts(marker = img)
            img_dir = img_mapping[img_file_key]
            file = find_file_by_name(files_path+img_dir, img_file_key+'_'+img_name)  # TODO Kostyl
            img_list.append(file[0].replace('\\', '/'))
    # IMAGES END

    main_response = clean_message_text(main_response)
    
    # Combine into final message
    if formatted_sources:
        sources_section = "\n\nSources:\n" + "\n".join(formatted_sources)
        telegram_message = main_response + sources_section
    else:
        telegram_message = main_response
    
    # Return dictionary with message and images
    return {
        "message": telegram_message,
        "images": img_list
    }

logging.basicConfig(
    level=logging.DEBUG, 
    filename='bot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TelegramBot:
    bot : telebot.TeleBot

    def __init__(self, bot_token:str) -> None:
        self.bot = telebot.TeleBot(token=bot_token)
        self.admin_messages = {}

        self.logger = logging.getLogger(__name__)

        self.bot.add_custom_filter(custom_filters.StateFilter(self.bot))
        self.register_handlers()
        self.orchestrator = create_orchestrator()

    def run_bot(self):
        """Start both the bot and reminder system"""
        
        # Start bot polling
        self.logger.info("Starting bot...")
        try:
            self.bot.polling(non_stop=True, timeout=60, long_polling_timeout=90) # Increased timeout
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
        def handle_message(msg):
            if msg.text == "Hi":
                self.bot.send_message(msg.chat.id, "Hello!", parse_mode=ParseMode.MARKDOWN)
            else:

                try:
                    self.bot.set_message_reaction(
                        msg.chat.id,
                        msg.message_id,
                        [ReactionTypeEmoji('👍')],
                        is_big=False
                    )

                    result = process_telegram_message(
                        self.orchestrator, 
                        msg.text, 
                        telegram_user_id=msg.from_user.id
                    )

                    debug_msg = "🔧 DEBUG MESSAGE:\n\n" \
                        f"🤖 *Routed to:* {clean_message_text(result['assistant_used'])}\n" \
                        f"❓ *Reason:* {clean_message_text(result['routing_decision']['reason'])}\n" \
                        f"📊 *Session:* {clean_message_text(result['session_id'])}" 
                    
                    self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    result_formatted = format_telegram_message(result)

                    if len(result_formatted['images']) < 1:
                        self.bot.send_message(msg.chat.id, result_formatted["message"], parse_mode=ParseMode.MARKDOWN)
                    else:
                        if len(result_formatted["images"]) == 1:
                            # Send single image
                            with open(result_formatted["images"][0], 'rb') as photo:
                                self.bot.send_photo(msg.chat.id, photo, caption=result_formatted["message"])
                        else:
                            # Send multiple images as media group
                            from telebot.types import InputMediaPhoto
    
                            if len(result_formatted["images"]) == 1:
                                with open(result_formatted["images"][0], 'rb') as photo:
                                    self.bot.send_photo(msg.chat.id, photo)
                            else:
                                # Keep files open during the entire operation
                                opened_files = []
                                media_group = []
                                
                                try:
                                    for i, img_path in enumerate(result_formatted["images"]):
                                        photo = open(img_path, 'rb')
                                        opened_files.append(photo)
                                        
                                        caption = result_formatted["message"] if i == 0 else None
                                        # Pass the file object, NOT the path string
                                        media_group.append(InputMediaPhoto(photo, caption=caption))
                                    
                                    self.bot.send_media_group(msg.chat.id, media_group)
                                    
                                finally:
                                    for photo in opened_files:
                                        photo.close()

                except Exception as e:
                    self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}', parse_mode=ParseMode.MARKDOWN)

                    if DEBUG:
                        self.bot.send_message(msg.chat.id, f'DEBUG MODE IS ACTIVE\n\nPlain text:\n{result_formatted["message"]}')

tg_bot = TelegramBot(bot_token=str(telegram_token))
tg_bot.run_bot()
