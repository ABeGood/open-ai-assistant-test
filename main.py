import os
from dotenv import load_dotenv
import re

from openai import OpenAI


load_dotenv()
openai_token = os.environ.get("OPENAI_TOKEN")
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

client = OpenAI(api_key=openai_token)
assistant = client.beta.assistants.retrieve(assistant_id='asst_OMuFH5IL2MXa4mEsabxBge3y')



import logging
import telebot # telebot
from datetime import datetime as dt, timedelta
import json
from time import sleep
from telebot import custom_filters, TeleBot
from telebot.types import ReactionTypeEmoji
from telegram.constants import ParseMode
from telebot.apihelper import ApiTelegramException
import re

logging.basicConfig(
    level=logging.DEBUG, 
    filename='bot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_message_text(text):
    """Clean text by removing problematic characters"""
    # Remove backslashes
    text = text.replace("\\", "")
    
    # Remove other potentially problematic characters
    problematic_chars = ['*', '_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in problematic_chars:
        text = text.replace(char, "")
    
    # Clean up whitespace
    text = " ".join(text.split())
    
    return text


thread = client.beta.threads.create()


class TelegramBot:
    bot : telebot.TeleBot

    def __init__(self, bot_token:str) -> None:
        self.bot = telebot.TeleBot(token=bot_token)
        self.admin_messages = {}

        self.logger = logging.getLogger(__name__)

        self.bot.add_custom_filter(custom_filters.StateFilter(self.bot))
        self.register_handlers()

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

        # Add new message handler for admin chat
        @self.bot.message_handler(
            func=lambda msg: msg.text is not None and '/' not in msg.text,
        )
        def handle_message(msg):
            if msg.text == "Hi":
                self.bot.send_message(msg.chat.id, "Hello!", parse_mode=ParseMode.MARKDOWN)
            else:

                self.bot.set_message_reaction(
                    msg.chat.id,
                    msg.message_id,
                    [ReactionTypeEmoji('👍')],
                    is_big=True
                )

                # sleep(0.2)
                # self.bot.send_message(
                #     msg.chat.id,
                #     "Ищу нужную инфу",
                #     parse_mode=ParseMode.MARKDOWN
                # )

                
                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=msg.text
                )

                run = client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )

                if run.status == 'completed': 
                    print('Success!')
                    messages = client.beta.threads.messages.list(
                        thread_id=thread.id
                    )

                    response_text = messages.data[0].content[0].text.value
                    img_markers = get_all_markers_as_list(response_text)

                    pattern = r'【.*?】'
                    response_text = re.sub(pattern, '', response_text)

                    if img_markers: # Only proceed if there are markers to delete
                        for marker in img_markers:
                            response_text = response_text.replace(marker, "")

                    responce_sorces = messages.data[0].content[0].text.annotations

                    sources_list = []
                    for source in responce_sorces:
                        file_info = client.files.retrieve(source.file_citation.file_id)
                        sources_list.append(f'files-ENG/eng-naming/{file_info.filename.split('.')[0].replace('_', '-')}.docx')
                        print(f"Filename: {file_info.filename}")

                    img_list = []
                    if img_markers:
                        for img in img_markers:
                            img_list.append(f'files_preproc/{file_info.filename.split('.')[0]}_IMG/{img.replace('@', '')}.png')

                    try:
                        if img_list:
                            with open(img_list[0], 'rb') as img_file:
                                # For photos, Telegram allows sending by file object or path
                                # Using file object for consistency with documents
                                self.bot.send_photo(
                                    chat_id=msg.chat.id,
                                    photo=img_file,
                                    caption=response_text + f"\n\nSources:\n{"\n".join(str(p) for p in sources_list)}",
                                )
                        else:
                            self.bot.send_message(
                                msg.chat.id,
                                response_text + f"\n\nSources:\n{"\n".join(str(p) for p in sources_list)}",
                                parse_mode=ParseMode.MARKDOWN
                            )
                    except ApiTelegramException as e:
                        self.bot.send_message(
                            msg.chat.id,
                            "Не получилось распарсить Markdown, отправляю как Plain Text",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        self.bot.send_message(
                            msg.chat.id,
                            clean_message_text(messages.data[0].content[0].text.value)
                        )
                    except Exception as e:
                        print(e)
                else:
                    print(run.status)
                    self.bot.send_message(
                        msg.chat.id,
                        "Что-то отвалилось :(",
                        parse_mode=ParseMode.MARKDOWN
                    )


tg_bot = TelegramBot(bot_token=str(telegram_token))
tg_bot.run_bot()
