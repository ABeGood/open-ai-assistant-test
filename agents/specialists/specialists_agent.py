from openai import OpenAI
import json
import time
import os
import uuid
import asyncio
import logging
from typing import Optional
from classes.agents_response_models import (
    SpecialistResponse,
    MultiSpecialistResponse,
    ResponseStatus,
    create_error_response,
    create_success_response,
    create_timeout_response
)
from agents.agents_config import assistant_configs
from agents.agent_response_processing_utils import (
    process_image_markers, 
    delete_sources_from_text, 
    extract_marker_parts,
    find_file_by_name
)
from agents.path_utils import (
    get_pdf_mapping_file_path,
    get_doc_mapping_file_path,
)
from config.paths import get_data_path_str
from classes.enums import SpecialistType

logging.basicConfig(
    level=logging.INFO, 
    filename='bot.log', 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class SpecialistAgent:
    """Agent for handling specialist assistant calls"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.assistants = {}
        self.shared_threads = {}
        self.context_store = {}
        self._initialize_assistants()

    def _initialize_assistants(self):
        """Initialize all registered assistants"""

        for name, config in assistant_configs.items():
            try:
                self.assistants[name] = {
                    'id': config['id'],
                    'purpose': config["purpose"],
                    'truncation_strategy': config["truncation_strategy"],
                    'max_prompt_tokens': config["max_prompt_tokens"],
                    'max_completion_tokens':config["max_completion_tokens"],
                    'assistant_obj': self.client.beta.assistants.retrieve(config['id'])
                }
            except Exception as e:
                print(f"Warning: Could not register assistant {name}: {e}")

    def create_session(self, telegram_user_id: Optional[int] = None) -> tuple[str, str]:
        """
        Create a session for user
        Returns: (session_id, message_for_user)
        """
        if telegram_user_id:
            session_id = f"tg-{telegram_user_id}"
            user_message = f"Session created for user {telegram_user_id}"
        else:
            session_id = f"anon_{uuid.uuid4().hex[:8]}"
            user_message = f"Anonymous session created: {session_id}"
        
        # Create thread
        thread = self.client.beta.threads.create()
        logging.info(f'Thread was created with id {thread.id}')
        self.shared_threads[session_id] = thread.id
        self.context_store[session_id] = {
            'conversation_history': [],
            'shared_context': {},
            'last_assistant': None,
            'routing_decisions': [],
            'telegram_user_id': telegram_user_id,
            'created_at': time.time()
        }
        
        return session_id, user_message
    
    def reset_context(self, session_id: str) -> str:
        """Reset context for a session while keeping the thread"""
        if session_id not in self.context_store:
            return "Session not found"
        
        # Keep telegram_user_id and created_at
        telegram_user_id = self.context_store[session_id].get('telegram_user_id')
        created_at = self.context_store[session_id].get('created_at')
        
        # Reset context but keep session
        self.context_store[session_id] = {
            'conversation_history': [],
            'shared_context': {},
            'last_assistant': None,
            'routing_decisions': [],
            'telegram_user_id': telegram_user_id,
            'created_at': created_at
        }
        
        return f"Context reset for session {session_id}"
    
    def delete_session(self, session_id: str) -> str:
        """Completely delete a session"""
        if session_id in self.shared_threads:
            del self.shared_threads[session_id]
        if session_id in self.context_store:
            del self.context_store[session_id]
        return f"Session {session_id} deleted"
    
    def get_or_create_session(self, telegram_user_id: Optional[int] = None) -> tuple[str, str]:
        """Get existing session or create new one"""
        if telegram_user_id:
            session_id = f"tg-{telegram_user_id}"
            if session_id in self.context_store:
                return session_id, f"Using existing session for user {telegram_user_id}"
        
        return self.create_session(telegram_user_id)

    async def _route_to_assistant_with_thread(self, 
                                              thread_id: str,
                                              assistant_name: str, 
                                              user_message: str) -> SpecialistResponse:
        """Route message to specific assistant with a specific thread ID"""
        start_time = time.time()
        
        try:
            assistant_id = self.assistants[assistant_name]['id']
            truncation_strategy_type = self.assistants[assistant_name]['truncation_strategy']['type']
            truncation_strategy_last_n = self.assistants[assistant_name]['truncation_strategy']['last_n_messages']
            max_prompt_tokens = self.assistants[assistant_name]['max_prompt_tokens']
            max_completion_tokens = self.assistants[assistant_name]['max_completion_tokens']
            
            # Run with specific assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                truncation_strategy={"type": truncation_strategy_type, "last_messages": truncation_strategy_last_n},
                max_prompt_tokens=max_prompt_tokens,
                max_completion_tokens=max_completion_tokens
            )
            
            # Wait for completion with timeout
            timeout = 60  # 60 seconds timeout
            run_start_time = time.time()
            
            while run.status != "completed":
                if time.time() - run_start_time > timeout:
                    return create_timeout_response(
                        SpecialistResponse,
                        user_message,
                        "Assistant response timed out",
                        specialist=assistant_name
                    )
                
                await asyncio.sleep(1)
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                
                if run.status == "failed":
                    return create_error_response(
                        SpecialistResponse,
                        f"Assistant run failed: {run.last_error}",
                        user_message,
                        specialist=assistant_name
                    )
            
            # Get response
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)

            # Parse the answer (Images, sources)
            sources_list = []
            response_sorces = messages.data[0].content[0].text.annotations
            for source in response_sorces:
                file_info = self.client.files.retrieve(source.file_citation.file_id)
                sources_list.append(file_info)

            text_wo_markers, img_markers_list = process_image_markers(messages.data[0].content[0].text.value)

            sources_files_list = []
            img_files_list = []

            if len(sources_list) > 0:
                try:
                    source_mapping_filepath = get_pdf_mapping_file_path(assistant_name)
                    with open(source_mapping_filepath, 'r', encoding='utf-8') as file:
                        pdf_mapping = json.load(file)
                    for source in sources_list:
                        source_filename = source.filename
                        name_without_ext = os.path.splitext(source_filename)[0]
                        source_file_path = pdf_mapping.get(name_without_ext)
                        if source_file_path:
                            sources_files_list.append(source_file_path)
                        else:
                            logging.warning(f'Source mapping not found for: {name_without_ext}')
                    sources_files_list = list(set(sources_files_list))
                except Exception as e:
                    logging.warning(f'Failed find source in pdf_mapping.json. {e}',exc_info=True)

            if len(img_markers_list) > 0:
                try:
                    img_mapping_filepath = get_doc_mapping_file_path(assistant_name)
                    with open(img_mapping_filepath, 'r', encoding='utf-8') as file:
                        img_mapping = json.load(file)
                    for img in img_markers_list:
                        img_info = extract_marker_parts(marker = img)
                        if img_info:
                            img_dir = img_mapping.get(img_info['img_file_key'])
                            if img_dir:
                                specialist_data_path = get_data_path_str(assistant_name)
                                file = find_file_by_name(os.path.join(specialist_data_path, img_dir), img_info['img_file_key']+'_'+img_info['img_name'])
                                if file:
                                    img_files_list.append(file[0].replace('\\', '/'))
                            else:
                                logging.warning(f'Image mapping not found for: {img_info["img_file_key"]}')
                    img_files_list = list(set(img_files_list))
                except Exception as e:
                    logging.warning(f'Failed to process image mappings. {e}', exc_info=True)

            response_clean = delete_sources_from_text(text_wo_markers)
            processing_time = time.time() - start_time

            return create_success_response(
                SpecialistResponse,
                user_message,
                specialist=assistant_name,
                response=response_clean,
                sources=sources_files_list,
                images=img_files_list,
                raw_response=messages.data[0].content[0].text.value,
                processing_time=processing_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                SpecialistResponse,
                f"Error processing response: {e}",
                user_message,
                specialist=assistant_name
            )

    def route_to_assistant(self, 
                           session_id: str, 
                           assistant_name: str, 
                           user_message: str, 
                           include_context: bool = True, 
                           new_thread: bool = False
                           ) -> SpecialistResponse:
        
        """Route message to specific assistant with context"""
        start_time = time.time()
        
        if session_id not in self.shared_threads:
            return create_error_response(
                SpecialistResponse,
                f"Session {session_id} not found",
                user_message,
                specialist=assistant_name
            )
        
        thread_id = self.shared_threads[session_id]
        assistant_id = self.assistants[assistant_name]['id']
        truncation_strategy_type = self.assistants[assistant_name]['truncation_strategy']['type']
        truncation_strategy_last_n = self.assistants[assistant_name]['truncation_strategy']['last_n_messages']
        max_prompt_tokens = self.assistants[assistant_name]['max_prompt_tokens']
        max_completion_tokens = self.assistants[assistant_name]['max_completion_tokens']
        
        # Run with specific assistant
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                truncation_strategy={"type": truncation_strategy_type, "last_messages": truncation_strategy_last_n},
                max_prompt_tokens=max_prompt_tokens,
                max_completion_tokens=max_completion_tokens
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                SpecialistResponse,
                f"Error creating run: {e}",
                user_message,
                specialist=assistant_name
            )
        
        # Wait for completion with timeout
        timeout = 60  # 60 seconds timeout
        run_start_time = time.time()
        
        while run.status != "completed":
            if time.time() - run_start_time > timeout:
                return create_timeout_response(
                    SpecialistResponse,
                    user_message,
                    "Assistant response timed out",
                    specialist=assistant_name
                )
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                return create_error_response(
                    SpecialistResponse,
                    f"Error retrieving run status: {e}",
                    user_message,
                    specialist=assistant_name
                )
            
            if run.status == "failed":
                return create_error_response(
                    SpecialistResponse,
                    f"Assistant run failed: {run.last_error}",
                    user_message,
                    specialist=assistant_name
                )
        
        # Get response
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)

            # TODO: Parse the answer (Images, sources)
            sources_list = []
            response_sorces = messages.data[0].content[0].text.annotations
            for source in response_sorces:
                file_info = self.client.files.retrieve(source.file_citation.file_id)
                sources_list.append(file_info)

            text_wo_markers, img_markers_list = process_image_markers(messages.data[0].content[0].text.value)

            sources_files_list = []
            img_files_list = []

            if len(sources_list) > 0:
                try:
                    source_mapping_filepath = get_pdf_mapping_file_path(assistant_name)
                    with open(source_mapping_filepath, 'r', encoding='utf-8') as file:
                        pdf_mapping = json.load(file)
                    for source in sources_list:
                        source_filename = source.filename
                        name_without_ext = os.path.splitext(source_filename)[0]
                        source_file_path = pdf_mapping.get(name_without_ext)
                        if source_file_path:
                            sources_files_list.append(source_file_path)
                        else:
                            logging.warning(f'Source mapping not found for: {name_without_ext}')
                    sources_files_list = list(set(sources_files_list))
                except Exception as e:
                    logging.warning(f'Failed find source in pdf_mapping.json. {e}',exc_info=True)

            if len(img_markers_list) > 0:
                try:
                    img_mapping_filepath = get_doc_mapping_file_path(assistant_name)
                    with open(img_mapping_filepath, 'r', encoding='utf-8') as file:
                        img_mapping = json.load(file)
                    for img in img_markers_list:
                        img_info = extract_marker_parts(marker = img)
                        if img_info:
                            img_dir = img_mapping.get(img_info['img_file_key'])
                            if img_dir:
                                specialist_data_path = get_data_path_str(assistant_name)
                                file = find_file_by_name(os.path.join(specialist_data_path, img_dir), img_info['img_file_key']+'_'+img_info['img_name'])
                                if file:
                                    img_files_list.append(file[0].replace('\\', '/'))
                            else:
                                logging.warning(f'Image mapping not found for: {img_info["img_file_key"]}')
                    img_files_list = list(set(img_files_list))
                except Exception as e:
                    logging.warning(f'Failed to process image mappings. {e}', exc_info=True)

            response_clean = delete_sources_from_text(text_wo_markers)
            processing_time = time.time() - start_time

        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                SpecialistResponse,
                f"Error processing response: {e}",
                user_message,
                specialist=assistant_name
            )
        
        response = create_success_response(
            SpecialistResponse,
            user_message,
            specialist=assistant_name,
            response=response_clean,
            sources=sources_files_list,
            images=img_files_list,
            raw_response=messages.data[0].content[0].text.value,
            processing_time=processing_time,
            timestamp=time.time()
        )
        
        # Update context
        if session_id in self.context_store:
            self.context_store[session_id]['last_assistant'] = assistant_name
            self.context_store[session_id]['conversation_history'].append({
                'timestamp': time.time(),
                'assistant': assistant_name,
                'role': 'assistant',
                'content': response.dict(),
                'message_type': 'response'
            })
        return response
    
    async def route_to_assistant_async(self, 
                                       session_id: str, 
                                       assistant_name: str, 
                                       user_message: str, 
                                       include_context: bool = True, 
                                       new_thread: bool = True) -> SpecialistResponse:
        """Async version of route_to_assistant"""
        # Convert your existing route_to_assistant to async
        # This depends on your OpenAI client implementation
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.route_to_assistant, 
            session_id, 
            assistant_name, 
            user_message, 
            include_context, 
            new_thread 
        )
    
    def call_specialists_sequentially(self, session_id: str, specialists_names: list[str], user_message: str) -> MultiSpecialistResponse:
        if SpecialistType.TABLES in specialists_names:
            specialists_names.remove(SpecialistType.TABLES)
        specialist_responses = []
        for specialist in specialists_names:
            response = self.route_to_assistant(
                session_id=session_id,
                assistant_name=specialist,
                user_message=user_message,
                include_context=False,
                new_thread=False
            )
            specialist_responses.append(response)
            
        # Process responses
        successful_responses = []
        failed_responses = []

        for response in specialist_responses:
            if response.success:
                successful_responses.append(response)
            else:
                failed_responses.append(response)

        success_rate = len(successful_responses) / len(specialist_responses) if specialist_responses else 0.0
        overall_success = len(successful_responses) > 0

        return MultiSpecialistResponse(
            success=overall_success,
            status=ResponseStatus.SUCCESS if overall_success else ResponseStatus.ERROR,
            error=None if overall_success else "All specialist calls failed",
            user_query=user_message,
            successful_responses=successful_responses,
            failed_responses=failed_responses,
            total_specialists=len(specialists_names),
            success_rate=success_rate,
            timestamp=time.time()
        )


    async def call_specialists_parallel(self, specialists_names: list[SpecialistType], user_message: str) -> MultiSpecialistResponse:
        # Create async tasks for all specialists with separate threads
        if SpecialistType.TABLES in specialists_names:
            specialists_names.remove(SpecialistType.TABLES)
            
        async def call_single_assistant(specialist_name: str):
            """Async wrapper for single assistant call with separate thread"""
            # Create a new thread for this specialist call
            thread = self.client.beta.threads.create()
            temp_thread_id = thread.id
            
            try:
                # Add the user message to the new thread
                self.client.beta.threads.messages.create(
                    thread_id=temp_thread_id,
                    role="user",
                    content=user_message
                )
                
                # Call the specialist using the new thread
                result = await self._route_to_assistant_with_thread(
                    thread_id=temp_thread_id,
                    assistant_name=specialist_name,
                    user_message=user_message
                )
                
                return result
                
            finally:
                # Clean up the thread after use
                try:
                    self.client.beta.threads.delete(temp_thread_id)
                except Exception as e:
                    logging.warning(f"Failed to delete thread {temp_thread_id}: {e}")
        
        # Execute all assistant calls in parallel
        tasks = [call_single_assistant(specialist) for specialist in specialists_names]

        try:
            specialist_responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses
            successful_responses = []
            failed_responses = []

            for response in specialist_responses:
                if isinstance(response, Exception):
                    # Convert exception to failed SpecialistResponse
                    failed_response = create_error_response(
                        SpecialistResponse,
                        str(response),
                        user_message,
                        specialist="unknown"
                    )
                    failed_responses.append(failed_response)
                elif response.success:
                    successful_responses.append(response)
                else:
                    failed_responses.append(response)

            success_rate = len(successful_responses) / len(specialist_responses) if specialist_responses else 0.0
            overall_success = len(successful_responses) > 0

            return MultiSpecialistResponse(
                success=overall_success,
                status=ResponseStatus.SUCCESS if overall_success else ResponseStatus.ERROR,
                error=None if overall_success else "All specialist calls failed",
                user_query=user_message,
                successful_responses=successful_responses,
                failed_responses=failed_responses,
                total_specialists=len(specialists_names),
                success_rate=success_rate,
                timestamp=time.time()
            )


        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return MultiSpecialistResponse(
                success=False,
                status=ResponseStatus.ERROR,
                error=f"Parallel execution failed: {e}",
                user_query=user_message,
                successful_responses=[],
                failed_responses=[],
                total_specialists=len(specialists_names),
                success_rate=0.0,
                timestamp=time.time()
            )

# Convenience functions for easy integration
def create_specialist_agent(api_key: str = None) -> SpecialistAgent:
    """Create and return configured specialist agent"""
    if api_key is None:
        api_key = os.environ.get("OPENAI_TOKEN")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_TOKEN environment variable not set")
    
    return SpecialistAgent(api_key)