"""
Telegram bot implementation for the customer support system.
"""

import os
import logging
import asyncio
from time import sleep
from typing import Set

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
from agents.orchestrator.orchestrator_agent import OrchestratorAgent
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
from .chat_utils import create_reaction_keyboard, create_locked_reaction_keyboard, parse_reaction_callback, get_reaction_emoji
from db_utils.db_manager import get_message_reaction, update_message_reaction

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

    def __init__(self, bot_token: str, llm_client:OpenAI) -> None:
        """
        Initialize the Telegram bot.
        
        Args:
            bot_token (str): Telegram bot token
            orchestrator: Orchestrator instance for processing queries
        """
        self.bot = AsyncTeleBot(token=bot_token)
        self.admin_messages = {}
        self.orchestrator_agent:OrchestratorAgent = OrchestratorAgent(llm_client=llm_client)
        self.table_agent:TableAgent = TableAgent(client=llm_client,
            prompt_strategy='hybrid_code_text',
            data_specs_dir_path=get_table_annotations_path(),
            generated_code_exec_timeout=60
        )
        self.logger = logging.getLogger(__name__)
        self.frozen_reaction_messages: Set[int] = set() # Track frozen reaction messages to prevent spam
        
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
                keyboard = create_reaction_keyboard(0)  # Will be updated after sending
                sent = await self.bot.send_message(msg.chat.id, "Hello!", parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
                message_id = getattr(sent, "message_id", None)
                
                test_msg = Message(
                    author="assistant",
                    content="Hello!",
                    message_id=message_id,
                    chat_id=tg_chat_id,
                )
                user.persist_append_messages([test_msg])
                
                if message_id: # Update keyboard with correct message_id
                    updated_keyboard = create_reaction_keyboard(message_id)
                    await self.bot.edit_message_reply_markup(msg.chat.id, message_id, reply_markup=updated_keyboard)
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

                    last_messages = user.get_chat_history(last_n=10)
                    orchestrator_response = self.orchestrator_agent.route_query_chat_completion(user_message, last_messages)

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
                                table_response_results[table] = {'response': resp, 'code': code[0].final_code_segment}
                                table_agent_interpreter_result = self.table_agent.interpret_result(
                                    user_query=user_message,
                                    code=table_response_results[table]['code'],
                                    output=table_response_results[table]['response']
                                    )
                                self.table_agent.agent_dataframe_manager.remove_all_data()
                    
                    specialists_responses = self.orchestrator_agent.call_specialists_sequentially(session_id=session_id, 
                                                                                          specialists_names=chosen_specialists.copy(),
                                                                                          user_message=user_message)
                    # Add table agent results to specialists_responses KOSTYL
                    if 'table_agent_interpreter_result' in locals():
                        if table_agent_interpreter_result.success:
                            specialists_responses.successful_responses.append(table_agent_interpreter_result)
                        else:
                            specialists_responses.failed_responses.append(table_agent_interpreter_result)
                    # Update totals and success rate
                    specialists_responses.total_specialists += 1
                    total_successful = len(specialists_responses.successful_responses)
                    specialists_responses.success_rate = total_successful /specialists_responses.total_specialists
                    specialists_responses.success = total_successful > 0

                    successfull_spec_resps = specialists_responses.successful_responses
                    failed_spec_resps = specialists_responses.failed_responses
                    
                    if DEBUG:
                        debug_msg = "🤖 STEP 2.2 \nSpecialists responses\n\n"\
                            f"⬅️ Получили ответы от специалистов:\n\n" \
                            f"{chosen_specialists}\n\n"\
                            f"Успешные: {len(successfull_spec_resps)}\n"\
                            f"Неуспешные: {len(failed_spec_resps)}"
                        await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    if len(successfull_spec_resps) == 1:
                        if DEBUG:
                            debug_msg = "📄 STEP 3 \nFormatting final response\n\n"\
                                f"Был получен один успешный ответ, оформляем его в финальное сообщение..."
                            await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)
                        final_answer_dict = successfull_spec_resps[0]
                    elif len(successfull_spec_resps) >1:
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
                    else:
                        if DEBUG:
                            debug_msg = "⚠️ Что-то пошло не так:\n\n"\
                                "Не было получено ни одного ответа."
                            await self.bot.send_message(msg.chat.id, debug_msg, parse_mode=ParseMode.MARKDOWN)

                    if final_answer_dict:
                        tg_message, images = format_telegram_message(final_answer_dict)

                    await self._send_response_with_images(msg.chat.id, tg_message, images, user, tg_chat_id)

                except Exception as e:
                    await self.bot.send_message(msg.chat.id, f'Что-то отвалилось :(\n\n{e}')

                    if DEBUG:
                        try:
                            await self.bot.send_message(msg.chat.id, f'DEBUG MODE IS ACTIVE\n\nPlain text:\n{tg_message}')
                        except Exception:
                            pass
        
        @self.bot.callback_query_handler(func=lambda callback: callback.data.startswith(("like:", "dislike:", "locked:")))
        async def handle_reaction_callback(reaction_callback):
            try:
                # Parse callback data
                reaction_type, message_id = parse_reaction_callback(reaction_callback.data)
                user_id = str(reaction_callback.from_user.id)
                
                # Check if message is frozen (being processed)
                if message_id in self.frozen_reaction_messages:
                    await self.bot.answer_callback_query(
                        reaction_callback.id, 
                        "Please wait. The previous request is being processed."
                    )
                    return
                
                # Handle locked callback (user clicked on disabled buttons)
                if reaction_type == "locked":
                    await self.bot.answer_callback_query(
                        reaction_callback.id, 
                        "Buttons are locked. Please wait."
                    )
                    return
                
                # Freeze the message to prevent spam
                self.frozen_reaction_messages.add(message_id)
                
                try:
                    # Lock the keyboard immediately
                    locked_keyboard = create_locked_reaction_keyboard(message_id)
                    await self.bot.edit_message_reply_markup(
                        reaction_callback.message.chat.id,
                        message_id,
                        reply_markup=locked_keyboard
                    )
                    
                    # Get current reaction for this message
                    current_reaction = get_message_reaction(user_id, message_id)
                    
                    # Determine new reaction
                    new_reaction = None
                    if reaction_type == "like":
                        new_reaction = Reaction.LIKE if current_reaction != Reaction.LIKE else None
                    elif reaction_type == "dislike":
                        new_reaction = Reaction.DISLIKE if current_reaction != Reaction.DISLIKE else None
                    
                    # Update reaction in database
                    update_message_reaction(user_id, message_id, new_reaction)
                    
                    # Update bot's reaction on the message
                    if new_reaction is None:
                        # Remove reaction
                        await self.bot.set_message_reaction(
                            reaction_callback.message.chat.id,
                            message_id,
                            []
                        )
                    else:
                        # Set reaction
                        emoji = get_reaction_emoji(new_reaction)
                        if emoji:
                            await self.bot.set_message_reaction(
                                reaction_callback.message.chat.id,
                                message_id,
                                [ReactionTypeEmoji(emoji)],
                                is_big=False
                            )
                    
                    # Restore normal keyboard after successful operation
                    normal_keyboard = create_reaction_keyboard(message_id)
                    await self.bot.edit_message_reply_markup(
                        reaction_callback.message.chat.id,
                        message_id,
                        reply_markup=normal_keyboard
                    )
                    
                    # Answer callback query to remove loading state
                    await self.bot.answer_callback_query(reaction_callback.id)
                    
                except Exception as e:
                    self.logger.error(f"Error processing reaction: {e}")
                    
                    # Show error message to user
                    await self.bot.answer_callback_query(
                        reaction_callback.id, 
                        f"Error while processing reaction: {str(e)}"
                    )
                    
                    # Keep keyboard locked on error (as requested)
                    # Don't restore normal keyboard, leave it locked
                    
                finally:
                    # Always unfreeze the message
                    self.frozen_reaction_messages.discard(message_id)
                
            except Exception as e:
                self.logger.error(f"Error handling reaction callback: {e}")
                await self.bot.answer_callback_query(
                    reaction_callback.id, 
                    "Error while processing request"
                )
                # Ensure message is unfrozen even on error
                if 'message_id' in locals():
                    self.frozen_reaction_messages.discard(message_id)

    async def _send_response_with_images(self, chat_id: int, message: str, images: list, user: User, tg_chat_id: int):
        """Send response message with optional images."""
        CAPTION_LIMIT = 1024
        MEDIA_GROUP_LIMIT = 10
        MESSAGE_LIMIT = 4096

        if len(images) < 1:
            # No images - just send text message
            if len(message) <= MESSAGE_LIMIT:
                keyboard = create_reaction_keyboard(0)  # Will be updated after sending
                sent = await self.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
                message_id = getattr(sent, "message_id", None)
                self._persist_message(user, message, message_id, tg_chat_id)
                # Update keyboard with correct message_id
                if message_id:
                    updated_keyboard = create_reaction_keyboard(message_id)
                    await self.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=updated_keyboard)
            else:
                # Split message into chunks by newlines
                chunks = []
                current_chunk = ""
                lines = message.split('\n')
                
                for line in lines:
                    # If adding this line would exceed limit, save current chunk
                    if len(current_chunk) + len(line) + 1 > MESSAGE_LIMIT:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = line
                        else:
                            # Single line too long, force split
                            chunks.append(line[:MESSAGE_LIMIT])
                            current_chunk = line[MESSAGE_LIMIT:]
                    else:
                        current_chunk += ('\n' if current_chunk else '') + line
                
                # Add remaining chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:  # Last chunk - add reaction keyboard
                        keyboard = create_reaction_keyboard(0)  # Will be updated after sending
                        sent = await self.bot.send_message(chat_id, chunk, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
                        message_id = getattr(sent, "message_id", None)
                        self._persist_message(user, message, message_id, tg_chat_id)
                        # Update keyboard with correct message_id
                        if message_id:
                            updated_keyboard = create_reaction_keyboard(message_id)
                            await self.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=updated_keyboard)
                    else:
                        sent = await self.bot.send_message(chat_id, chunk, parse_mode=ParseMode.MARKDOWN)
                        self._persist_message(user, message, getattr(sent, "message_id", None), tg_chat_id)
        
        elif len(images) == 1:
            caption_too_long = len(message) > CAPTION_LIMIT
            # Send single image
            with open(images[0], 'rb') as photo:
                if caption_too_long:
                    # Send image without caption, then text separately
                    await self.bot.send_photo(chat_id, photo)
                    keyboard = create_reaction_keyboard(0)  # Will be updated after sending
                    sent_text = await self.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
                    message_id = getattr(sent_text, "message_id", None)
                    self._persist_message(user, message, message_id, tg_chat_id)
                    
                    # Update keyboard with correct message_id
                    if message_id:
                        updated_keyboard = create_reaction_keyboard(message_id)
                        await self.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=updated_keyboard)
                else:
                    # Send image with caption
                    keyboard = create_reaction_keyboard(0)  # Will be updated after sending
                    sent_photo = await self.bot.send_photo(chat_id, photo, caption=message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
                    message_id = getattr(sent_photo, "message_id", None)
                    self._persist_message(user, message, message_id, tg_chat_id)
                    
                    # Update keyboard with correct message_id
                    if message_id:
                        updated_keyboard = create_reaction_keyboard(message_id)
                        await self.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=updated_keyboard)
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
            keyboard = create_reaction_keyboard(0)  # Will be updated after sending
            sent_caption = await self.bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
            assistant_msg_id_to_save = getattr(sent_caption, "message_id", None)
            
            # Update keyboard with correct message_id
            if assistant_msg_id_to_save:
                updated_keyboard = create_reaction_keyboard(assistant_msg_id_to_save)
                await self.bot.edit_message_reply_markup(chat_id, assistant_msg_id_to_save, reply_markup=updated_keyboard)
        else:
            assistant_msg_id_to_save = first_group_first_msg_id
            # Add keyboard to the first message of the media group
            if assistant_msg_id_to_save:
                keyboard = create_reaction_keyboard(assistant_msg_id_to_save)
                await self.bot.edit_message_reply_markup(chat_id, assistant_msg_id_to_save, reply_markup=keyboard)
               
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