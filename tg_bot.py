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


# def format_telegram_message(response_data):
#     """
#     Convert LLM agent response to Telegram message format
    
#     Args:
#         response_data (dict or str): LLM agent response dictionary or JSON string
    
#     Returns:
#         str: Formatted Telegram message
#     """
#     # Parse JSON string if needed
#     if isinstance(response_data, str):
#         response_data = json.loads(response_data)

#     # CATEGORY START

#     files_path = None
#     assistant_used = response_data['assistant_used']
#     if assistant_used == 'equipment':
#         files_path = 'files_preproc/equipment/'
#     elif assistant_used == 'cables':
#         files_path = 'files_preproc/cables/'
#     elif assistant_used == 'tools':
#         files_path = 'files_preproc/tools/'
#     elif assistant_used == 'commonInfo':
#         files_path = 'files_preproc/common_info/'

#     # CATEGORY END
    
#     # Extract main response text
#     # main_response = clean_message_text(response_data['response']['response'])
#     main_response = delete_sources_from_text(response_data['response']['response'])
    

#     # SOURCES START
#     # Extract and format sources
#     if files_path:
#         sources = response_data['response']['sources']
#         formatted_sources = []
    
#         for source in sources:
#             # Get filename without extension
#             filename = source.filename
#             name_without_ext = os.path.splitext(filename)[0]
            
#             try:
#                 source_mapping_filepath = files_path + 'pdf_mapping.json'
#                 with open(source_mapping_filepath, 'r', encoding='utf-8') as file:
#                     pdf_mapping = json.load(file)
#                 link = pdf_mapping[name_without_ext]
#                 # Create markdown link
#                 markdown_link = f"[{name_without_ext.split('_')[0]}]({link})"
#                 formatted_sources.append(markdown_link)
#                 formatted_sources = list(set(formatted_sources))
#             except Exception as e:
#                 markdown_link = f"{filename}"
#                 formatted_sources.append(markdown_link)
#                 formatted_sources = list(set(formatted_sources))

#     # SOURCES END


#     # IMAGES START
#     img_markers = get_all_markers_as_list(main_response)
#     if img_markers and files_path:
#         img_mapping_filepath = files_path + 'doc_mapping.json'
#         with open(img_mapping_filepath, 'r', encoding='utf-8') as file:
#             img_mapping = json.load(file)
        
#         img_list = []
#         for img in img_markers:
#             img_file_key, img_name = extract_marker_parts(marker = img)
#             img_dir = img_mapping[img_file_key]
#             file = find_file_by_name(files_path+img_dir, img_file_key+'_'+img_name)  # TODO Kostyl
#             img_list.append(file[0].replace('\\', '/'))
#     # IMAGES END

#     main_response = clean_message_text(main_response)
    
#     # Combine into final message
#     if formatted_sources:
#         sources_section = "\n\nSources:\n" + "\n".join(formatted_sources)
#         telegram_message = main_response + sources_section
#     else:
#         telegram_message = main_response
    
#     # Return dictionary with message and images
#     return {
#         "message": telegram_message,
#         "images": img_list
#     }

logging.basicConfig(
    level=logging.DEBUG, 
    filename='bot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
                        f"🤖 *Routed to:* {clean_message_text(result['assistant_used'])}\n" \
                        f"❓ *Reason:* {clean_message_text(result['routing_decision']['reason'])}\n" \
                        f"📊 *Session:* {clean_message_text(result['session_id'])}" 
                    
                    await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    result_formatted = format_telegram_message(result)

                    if len(result_formatted['images']) < 1:
                        await self.bot.send_message(msg.chat.id, result_formatted["message"], parse_mode=ParseMode.MARKDOWN)
                    else:
                        if len(result_formatted["images"]) == 1:
                            # Send single image
                            with open(result_formatted["images"][0], 'rb') as photo:
                                await self.bot.send_photo(msg.chat.id, photo, caption=result_formatted["message"])
                        else:
                            # Send multiple images as media group
                            from telebot.types import InputMediaPhoto
    
                            if len(result_formatted["images"]) == 1:
                                with open(result_formatted["images"][0], 'rb') as photo:
                                    await self.bot.send_photo(msg.chat.id, photo)
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
                                    
                                    await self.bot.send_media_group(msg.chat.id, media_group)
                                    
                                finally:
                                    for photo in opened_files:
                                        photo.close()

                except Exception as e:
                    # await self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}', parse_mode=ParseMode.MARKDOWN)
                    await self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}')

                    if DEBUG:
                        await self.bot.send_message(msg.chat.id, f'DEBUG MODE IS ACTIVE\n\nPlain text:\n{result_formatted["message"]}')

tg_bot = TelegramBot(bot_token=str(telegram_token))
tg_bot.run_bot()
