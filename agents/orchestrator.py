from openai import OpenAI
import json
import time
import os
import uuid
import asyncio
import logging
from typing import Optional
from classes.validators import OrchestratorResponse as ValidatorOrchestratorResponse
from classes.agents_response_models import (
    OrchestratorResponse,
    SpecialistResponse,
    CombinatorResponse,
    MultiSpecialistResponse,
    ResponseStatus,
    create_error_response,
    create_success_response,
    create_timeout_response
)
from agents.prompt_static_analyzer.prompt_static_analyzer import PromptStaticAnalyzer
from agents.prompt_static_analyzer.formatters import format_analyzer_output_for_orchestrator
from agents.agents_config import assistant_configs, price_per_token_in, price_per_token_out
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


class OrchestratorAgent:
    """Multi-assistant orchestrator optimized for Telegram bot integration"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.assistants = {}
        self.shared_threads = {}
        self.context_store = {}
        self._initialize_assistants()
        self.static_checker = PromptStaticAnalyzer()


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


    def process_with_orchestrator(self, session_id: str, user_message: str) -> OrchestratorResponse:
        """Use orchestrator assistant to determine routing"""
        if session_id not in self.shared_threads:
            logging.error(f"Error occurred: session with id {session_id} not found.")
            return create_error_response(
                OrchestratorResponse,
                "Session not found",
                user_message,
                specialists=[],
                reason=None,
                tables_to_query = [],
            )
        
        # Get current context
        # context = self.get_shared_context(session_id)
        user_message_metadata = self.static_checker.route_query(user_message)
        formatted_user_msg_metadata = format_analyzer_output_for_orchestrator(user_message_metadata)
        
        # Create routing prompt
        routing_prompt = f"""
USER REQUEST: 
{user_message}

USER REQUEST METADATA FROM STATIC ANALYSIS:
{formatted_user_msg_metadata}
"""
        
        # Route to orchestrator for decision
        thread_id = self.shared_threads[session_id]
        orchestrator_id = self.assistants['orchestrator']['id']
        truncation_strategy_type = self.assistants['orchestrator']['truncation_strategy']['type']
        truncation_strategy_last_n = self.assistants['orchestrator']['truncation_strategy']['last_n_messages']
        max_prompt_tokens = self.assistants['orchestrator']['max_prompt_tokens']
        max_completion_tokens = self.assistants['orchestrator']['max_completion_tokens']
        
        # Add routing message
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=routing_prompt,
                metadata={"type": "routing_decision"}
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                OrchestratorResponse,
                f"Error creating routing message: {e}",
                user_message,
                specialists=[],
                tables_to_query = [],
            )
        
        # Get routing decision
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=orchestrator_id,
                truncation_strategy={"type": truncation_strategy_type, "last_messages": truncation_strategy_last_n},
                max_prompt_tokens=max_prompt_tokens,
                max_completion_tokens=max_completion_tokens
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                OrchestratorResponse,
                f"Error creating routing run: {e}",
                user_message,
                specialists=[],
                tables_to_query = [],
            )
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return create_timeout_response(
                    OrchestratorResponse,
                    user_message,
                    "Routing decision timed out",
                    specialists=[],
                    tables_to_query = [],
                )
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                return create_error_response(
                    OrchestratorResponse,
                    f"Error checking routing status: {e}",
                    user_message,
                    specialists=[],
                    tables_to_query = [],
                )
        logging.info(f'ORCHESTRATOR RUN COMPLETED:\nInput tokens: {run.usage.prompt_tokens} ({run.usage.prompt_tokens*price_per_token_in}$)\nInput tokens: {run.usage.completion_tokens} ({run.usage.completion_tokens*price_per_token_out}$)')
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            routing_decision_raw = messages.data[0].content[0].text.value
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                OrchestratorResponse,
                f"Error retrieving routing decision: {e}",
                user_message,
                specialists=[],
                tables_to_query = [],
            )
        
        # Parse decision - try to extract JSON from response
        try:
            # Look for JSON in the response
            ValidatorOrchestratorResponse.model_validate_json(routing_decision_raw)
            orchestrator_response_dict = json.loads(routing_decision_raw)

            if len(orchestrator_response_dict['specialists']) > 0:
                # Log routing decision
                if session_id in self.context_store:
                    self.context_store[session_id]['routing_decisions'].append(orchestrator_response_dict)
                
                return create_success_response(
                    OrchestratorResponse,
                    user_message,
                    tables_to_query = orchestrator_response_dict.get('tables_to_query', []),
                    specialists=orchestrator_response_dict.get('specialists', []),
                    reason=orchestrator_response_dict.get('reason'),
                    confidence=orchestrator_response_dict.get('confidence'),
                    raw_response=routing_decision_raw,
                    timestamp=time.time()
                )
            else:
                return create_error_response(
                    OrchestratorResponse,
                    "Empty assistants list",
                    user_message,
                    specialists=[],
                    tables_to_query = [],
                )
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                OrchestratorResponse,
                f"Error parsing orchestrator response: {e}",
                user_message,
                specialists=[],
                tables_to_query = [],
            )
    
    def process_with_combinator(self, session_id: str, user_message: str, specialists_responses: list[SpecialistResponse]) -> CombinatorResponse:
        """Use combinator assistant to prepare the final answer"""
        specialists_names = [resp.specialist for resp in specialists_responses]

        if session_id not in self.shared_threads:
            return create_error_response(
                CombinatorResponse,
                "Session not found",
                user_message,
                specialists=specialists_names
            )
        
        # Get current context
        # context = self.get_shared_context(session_id)
        
        # Create combinator prompt
        specialist_responses = ""
        for i, item in enumerate(specialists_responses):
            specialist_responses += f"SPECIALIST {i+1} DATA START\n"
            specialist_responses += f"SPECIALIST NAME: {item.specialist}\n"
            specialist_responses += f"SPECIALIST RESPONSE: {item.response}\n"
            specialist_responses += f"SPECIALIST {i+1} DATA END\n\n"

        combinator_prompt = f"""USER QUERY: 
{user_message}

SPECIALISTS RESPONSES:
{specialist_responses}
"""
        
        # Route to combinator for decision
        thread_id = self.shared_threads[session_id]
        combinator_id = self.assistants['combinator']['id']
        truncation_strategy_type = self.assistants['combinator']['truncation_strategy']['type']
        truncation_strategy_last_n = self.assistants['combinator']['truncation_strategy']['last_n_messages']
        max_prompt_tokens = self.assistants['combinator']['max_prompt_tokens']
        max_completion_tokens = self.assistants['combinator']['max_completion_tokens']
        
        # Add routing message
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="assistant",
                content=combinator_prompt,
                metadata={"type": "combinator_call"}
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                CombinatorResponse,
                f"Error creating routing message: {e}",
                user_message,
                specialists=specialists_names
            )
        
        # Get final answer
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=combinator_id,
                truncation_strategy={"type": truncation_strategy_type, "last_messages": truncation_strategy_last_n},
                max_prompt_tokens=max_prompt_tokens,
                max_completion_tokens=max_completion_tokens
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                CombinatorResponse,
                f"Error creating routing run: {e}",
                user_message,
                specialists=specialists_names
            )
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return create_timeout_response(
                    CombinatorResponse,
                    user_message,
                    "Combinator call timed out",
                    specialists=specialists_names
                )
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                return create_error_response(
                    CombinatorResponse,
                    f"Error checking combinator status: {e}",
                    user_message,
                    specialists=specialists_names
                )
        
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            final_response = messages.data[0].content[0].text.value

            # Extract sources
            sources_list = []
            for spec_response in specialists_responses:
                sources_list += spec_response.sources
                sources_list = list(set(sources_list))

            # Extract images
            img_list = []
            for spec_response in specialists_responses:
                img_list += spec_response.images
                img_list = list(set(img_list))

        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return create_error_response(
                CombinatorResponse,
                f"Error retrieving combinator response: {e}",
                user_message,
                specialists=specialists_names
            )
        
        return create_success_response(
            CombinatorResponse,
            user_message,
            specialists=specialists_names,
            response=final_response,
            sources=sources_list,
            images=img_list,
            raw_response=final_response,
            combined_from=specialists_responses,
            timestamp=time.time()
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
                                file = find_file_by_name(os.path.join(files_path, img_dir), img_info['img_file_key']+'_'+img_info['img_name'])
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
    
    async def process_request(self, session_id: str, user_message: str, telegram_user_id: Optional[int] = None) -> OrchestratorResponse:
        """Complete request processing with routing and context"""
        
        # Ensure session exists
        if session_id not in self.context_store:
            session_id, session_msg = self.get_or_create_session(telegram_user_id)
        
        # Determine which assistant to use
        return self.process_with_orchestrator(session_id, user_message)
    
    def call_specialists_sequentially(self, session_id: str, specialists_names: list[str], user_message: str) -> MultiSpecialistResponse:
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


    async def call_specialists_parallel(self, session_id: str, specialists_names: list[str], user_message: str) -> MultiSpecialistResponse:
        # Create async tasks for all specialists
        async def call_single_assistant(specialist_name: str):
            """Async wrapper for single assistant call"""
            return await self.route_to_assistant_async(
                session_id=session_id,
                assistant_name=specialist_name,
                user_message=user_message,
                include_context=True
            )
        
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
def create_orchestrator(api_key: str = None) -> OrchestratorAgent:
    """Create and return configured orchestrator"""
    if api_key is None:
        api_key = os.environ.get("OPENAI_TOKEN")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_TOKEN environment variable not set")
    
    return OrchestratorAgent(api_key)
