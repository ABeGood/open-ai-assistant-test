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

DEBUG = True

load_dotenv()
telegram_token = os.environ.get("TELEGRAM_TOKEN")

IMG_PATTERN = r"@_IMG_\d+"

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

    
    
    # Extract main response text
    main_response = clean_message_text(response_data['response']['response'])
    main_response = delete_sources_from_text(main_response)
    
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
    
    # Combine into final message
    if formatted_sources:
        sources_section = "\n\nSources:\n" + "\n".join(formatted_sources)
        telegram_message = main_response + sources_section
    else:
        telegram_message = main_response
    
    return telegram_message

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

                    result_msg = format_telegram_message(result)
                    
                    self.bot.send_message(msg.chat.id, result_msg, parse_mode=ParseMode.MARKDOWN)
                except Exception as e:
                    self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}', parse_mode=ParseMode.MARKDOWN)

                    if DEBUG:
                        self.bot.send_message(msg.chat.id, f'DEBUG MODE IS ACTIVE\n\nPlain text:\n{result_msg}')

tg_bot = TelegramBot(bot_token=str(telegram_token))
tg_bot.run_bot()
