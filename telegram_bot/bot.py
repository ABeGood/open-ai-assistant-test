"""
Telegram bot implementation for the customer support system.
"""

import os
import logging
import asyncio
from time import sleep

from dotenv import load_dotenv
from telebot import custom_filters
from telebot.async_telebot import AsyncTeleBot
from telebot.types import ReactionTypeEmoji
from telegram.constants import ParseMode
from telebot.apihelper import ApiTelegramException
from telebot.types import InputMediaPhoto

from classes.classes import User, Message, Reaction
from classes.enums import SpecialistType
from classes.agents_response_models import SpecialistResponse, CombinatorResponse
from .formatters import format_telegram_message
from agents.orchestrator import OrchestratorAgent
from agents.table_processor import TableAgent
from openai import OpenAI
from agents.path_utils import (
    resolve_image_path, 
    resolve_all_images_in_text, 
    find_specialist_files,
    get_specialist_base_path,
    get_pdf_mapping_file_path,
    get_doc_mapping_file_path,
    get_table_data_path,
    get_table_annotations_path
)

DEBUG = True

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(
    level=logging.DEBUG, 
    filename='bot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TelegramBot:
    """Telegram bot for handling customer support queries."""

    def __init__(self, bot_token: str, orchestrator:OrchestratorAgent, llm_client:OpenAI) -> None:
        """
        Initialize the Telegram bot.
        
        Args:
            bot_token (str): Telegram bot token
            orchestrator: Orchestrator instance for processing queries
        """
        self.bot = AsyncTeleBot(token=bot_token)
        self.admin_messages = {}
        self.orchestrator_agent:OrchestratorAgent = orchestrator
        self.table_agent:TableAgent = TableAgent(client=llm_client,
            prompt_strategy='hybrid_code_text',
            data_specs_dir_path=get_table_annotations_path(),
            generated_code_exec_timeout=60
        )
        self.logger = logging.getLogger(__name__)
        
        self.register_handlers()

    def run(self):
        """Start the bot polling."""
        self.logger.info("Starting bot...")
        try:
            asyncio.run(self.bot.polling(non_stop=True, timeout=60, request_timeout=90))
        except Exception as e:
            self.logger.error(f"Bot polling stopped due to an error: {e}")
            sleep(5)
            self.run()

    def register_handlers(self):
        """Register all message handlers."""

        @self.bot.message_handler(
            func=lambda msg: msg.text is not None and '/' not in msg.text,
        )
        async def handle_message(msg):
    
            telegram_user_id = msg.from_user.id
            session_id = f"tg-{telegram_user_id}"
            user_message = msg.text

            tg_chat_id = msg.chat.id
            user_name = getattr(msg.from_user, "full_name", None) or getattr(msg.from_user, "username", None) or "Unknown"

            # создадим (если нет) и инициализируем объект User
            user = User.ensure(user_id=str(telegram_user_id), name=user_name, cache_maxlen=200)

            # подгрузим последние n сообщений (для локального кеша в User)
            user.get_last_n_msgs_from_db(n=200)

            # сохраним текущее входящее от пользователя сообщение в БД + добавим в кеш
            user_msg = Message(
                author=telegram_user_id,
                content=user_message,
                message_id=getattr(msg, "message_id", None),
                chat_id=tg_chat_id,
            )
            user.persist_append_messages([user_msg])

            if msg.text == "Hi":
                await self.bot.send_message(msg.chat.id, "Hello!", parse_mode=ParseMode.MARKDOWN)
            else:

                try:
                    await self.bot.set_message_reaction(
                        msg.chat.id,
                        msg.message_id,
                        [ReactionTypeEmoji('👌')],
                        is_big=False
                    )

                    if DEBUG:
                        debug_msg = "🔀 STEP 1.1 \nOrchestrator call\n\n"\
                            f"Определяем каких ботов-специалистов использовать для этого запроса..."
                        await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    orchestrator_response = await self.orchestrator_agent.process_request(session_id, user_message, telegram_user_id)

                    if DEBUG:
                        debug_msg = "🔀 STEP 1.2 \nOrchestrator response\n\n"\
                            f"🤖 Выбранные специалисты: \n{', '.join(orchestrator_response.specialists)}\n\n" \
                            f"❓ Причина: \n{orchestrator_response.reason}".replace("_", "\\_")
                        await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    chosen_specialists = orchestrator_response.specialists

                    if len(chosen_specialists) < 1:
                        if DEBUG:
                            debug_msg = "⚠️ Что-то пошло не так:\n\n" \
                                "Не был выбран ни один специалист..."
                            await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)
                            raise Exception(f"No specialists were selected for query {user_message}")
                        
                    if DEBUG:
                        debug_msg = "🤖 STEP 2.1 \nSpecialists call\n\n"\
                            f"➡️ Отправляем параллельные запросы ботам-специалистам:\n\n" \
                            f"{chosen_specialists}"
                        await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    # Call table processor if needed
                    if SpecialistType.TABLES in chosen_specialists:
                        if orchestrator_response.tables_to_query:
                            table_response_results = {}
                            for table in orchestrator_response.tables_to_query:
                                table_file_path = get_table_data_path(table_name=table)
                                self.table_agent.agent_dataframe_manager.add_data(table_file_path)
                                resp, code = self.table_agent.answer_query(user_message)
                                table_response_results[table]['response'] = resp
                                table_response_results[table]['code'] = code
                                self.table_agent.agent_dataframe_manager.remove_all_data()
                    
                    specialists_responses = self.orchestrator_agent.call_specialists_sequentially(session_id=session_id, 
                                                                                          specialists_names=chosen_specialists,
                                                                                          user_message=user_message)
                    
                    successfull_spec_resps = specialists_responses.successful_responses
                    failed_spec_resps = specialists_responses.failed_responses
                    
                    if DEBUG:
                        debug_msg = "🤖 STEP 2.2 \nSpecialists responses\n\n"\
                            f"⬅️ Получили ответы от специалистов:\n\n" \
                            f"{chosen_specialists}\n\n"\
                            f"Успешные: {len(successfull_spec_resps)}\n"\
                            f"Неуспешные: {len(failed_spec_resps)}"
                        await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    if len(successfull_spec_resps) < 2:
                        if DEBUG:
                            debug_msg = "📄 STEP 3 \nFormatting final response\n\n"\
                                f"Был получен один успешный ответ, оформляем его в финальное сообщение..."
                            await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)
                            final_answer_dict = successfull_spec_resps[0]
                    else:
                        if DEBUG:
                            debug_msg = "🔗 STEP 3.1 \nCombinator call\n\n"\
                                f"Было получено несколько успешных ответов.\n\n"\
                                "➡️ Направляем их в бот-комбинатор для составления финального ответа."
                            await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)
                            
                            final_answer_dict = self.orchestrator_agent.process_with_combinator(session_id, user_message, successfull_spec_resps)

                            if DEBUG:
                                debug_msg = "🔗 STEP 3.2 \nCombinator response\n\n"\
                                    "⬅️ Получили ответ от комбинатора."
                                await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                                debug_msg = "📄 STEP 4 \nFormatting final response\n\n"\
                                f"Оформляем ответ комбинатора в финальное сообщение..."
                                await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    tg_message, images = format_telegram_message(final_answer_dict)

                    await self._send_response_with_images(msg.chat.id, tg_message, images, user, tg_chat_id)

                except Exception as e:
                    await self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}')

    async def _send_response_with_images(self, chat_id: int, message: str, images: list, user: User, tg_chat_id: int):
        """Send response message with optional images."""
        CAPTION_LIMIT = 1024
        MEDIA_GROUP_LIMIT = 10
        
        caption_too_long = len(message) > CAPTION_LIMIT

        if len(images) < 1:
            # No images - just send text message
            sent = await self.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN)
            self._persist_message(user, message, getattr(sent, "message_id", None), tg_chat_id)
        
        elif len(images) == 1:
            # Send single image
            with open(images[0], 'rb') as photo:
                if caption_too_long:
                    # Send image without caption, then text separately
                    await self.bot.send_photo(chat_id, photo)
                    sent_text = await self.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN)
                    self._persist_message(user, message, getattr(sent_text, "message_id", None), tg_chat_id)
                else:
                    # Send image with caption
                    sent_photo = await self.bot.send_photo(chat_id, photo, caption=message, parse_mode=ParseMode.MARKDOWN)
                    self._persist_message(user, message, getattr(sent_photo, "message_id", None), tg_chat_id)
        else:
            # Send multiple images
            await self._send_media_group(chat_id, images, message, caption_too_long, user, tg_chat_id)

    async def _send_media_group(self, chat_id: int, images: list, message: str, caption_too_long: bool, user: User, tg_chat_id: int):
        """Send multiple images as media groups."""
        MEDIA_GROUP_LIMIT = 10
        
        image_chunks = [images[i:i + MEDIA_GROUP_LIMIT] for i in range(0, len(images), MEDIA_GROUP_LIMIT)]
        first_group_first_msg_id = None
        
        for chunk_index, image_chunk in enumerate(image_chunks):
            opened_files = []
            media_group = []
            
            try:
                for i, img_path in enumerate(image_chunk):
                    photo = open(img_path, 'rb')
                    opened_files.append(photo)
                    
                    # Only add caption to first image of first chunk if it's not too long
                    caption = None
                    if chunk_index == 0 and i == 0 and not caption_too_long:
                        caption = message
                    
                    media_group.append(InputMediaPhoto(
                        photo, 
                        caption=caption, 
                        parse_mode=ParseMode.MARKDOWN if caption else None
                    ))
                
                group_messages = await self.bot.send_media_group(chat_id, media_group)

                # запомним id первой записи первой группы (если подпись помещается)
                if chunk_index == 0 and not caption_too_long and group_messages:
                    first_group_first_msg_id = getattr(group_messages[0], "message_id", None)
            finally:
                for photo in opened_files:
                    photo.close()
        
        # If caption was too long, send it as separate message after all media groups
        assistant_msg_id_to_save = None
        if caption_too_long:
            sent_caption = await self.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN)
            assistant_msg_id_to_save = getattr(sent_caption, "message_id", None)
        else:
            assistant_msg_id_to_save = first_group_first_msg_id
               
        self._persist_message(user, message, assistant_msg_id_to_save, tg_chat_id)

    def _persist_message(self, user: User, content: str, message_id: int, chat_id: int):
        """Persist assistant message to database."""
        try:
            reply = Message(
                author=f"bot_to_{user.get_user_id()}",
                content=content,
                message_id=message_id,
                chat_id=chat_id,
            )
            user.persist_append_messages([reply])
        except Exception as e:
            self.logger.error(f"Failed to save assistant message: {e}")