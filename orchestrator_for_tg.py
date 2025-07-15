from openai import OpenAI
import json
import time
import os
import uuid
import asyncio
import logging
from typing import Optional
from validators import OrchestratorResponse
from config import assistant_configs, price_per_token_in, price_per_token_out
from response_processing_utils import (
    process_image_markers, 
    delete_sources_from_text, 
    assistant_files_mapping, 
    extract_marker_parts,
    find_file_by_name
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


class TelegramMultiAssistantOrchestrator:
    """Multi-assistant orchestrator optimized for Telegram bot integration"""
    
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
    
    # def add_context_message(self, session_id: str, assistant_name: str, content: str, role: str = "assistant"):
    #     """Add contextual message to maintain consistency"""
    #     if session_id not in self.context_store:
    #         return
        
    #     thread_id = self.shared_threads[session_id]
        
    #     # Add to context store
    #     context_entry = {
    #         'timestamp': time.time(),
    #         'assistant': assistant_name,
    #         'role': role,
    #         'content': content,
    #         'message_type': 'context_update'
    #     }
        
    #     self.context_store[session_id]['conversation_history'].append(context_entry)
        
    #     # Add hidden context message to thread
    #     context_message = f"[CONTEXT UPDATE from {assistant_name}]: {content}"
        
    #     try:
    #         self.client.beta.threads.messages.create(
    #             thread_id=thread_id,
    #             role="assistant",
    #             content=context_message,
    #             metadata={"type": "context", "source_assistant": assistant_name}
    #         )
    #     except Exception as e:
    #         print(f"Error adding context message: {e}")
    
    # def get_shared_context(self, session_id: str) -> str:
    #     """Get formatted shared context for assistant"""
    #     if session_id not in self.context_store:
    #         return "=== NO CONTEXT ==="
        
    #     context = self.context_store[session_id]
        
    #     context_summary = "=== SHARED CONTEXT ===\n"
    #     context_summary += f"Previous assistants involved: {set(h['assistant'] for h in context['conversation_history'])}\n"
    #     context_summary += f"Last assistant: {context['last_assistant']}\n"
        
    #     # Recent conversation summary
    #     recent_history = context['conversation_history'][-5:]  # Last 5 interactions
    #     context_summary += "\nRECENT INTERACTIONS:\n"
        
    #     for entry in recent_history:
    #         if entry['role'] == 'user':
    #             context_summary += f"User: {entry['content']['response'][:100]}...\n"
    #         else:
    #             context_summary += f"{entry['assistant']}: {entry['content']['response'][:100]}...\n"
        
    #     # Shared data
    #     if context['shared_context']:
    #         context_summary += f"\nSHARED DATA: {json.dumps(context['shared_context'], indent=2)}\n"
        
    #     context_summary += "=== END CONTEXT ===\n"
        
    #     return context_summary

    def process_with_orchestrator(self, session_id: str, user_message: str) -> dict:
        """Use orchestrator assistant to determine routing"""
        if session_id not in self.shared_threads:
            logging.error(f"Error occurred: session with id {session_id} not found.")
            return {"success": False,
                    "error": "Session not found",
                    "specialists": [],
                    "reason": None,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Get current context
        # context = self.get_shared_context(session_id)
        
        # Create routing prompt
        routing_prompt = f"""
USER REQUEST: 
{user_message}
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
            return {"success": False,
                    "error": f"Error creating routing message: {e}",
                    "specialists": [],
                    "reason": None,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
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
            return {"success": False,
                    "error": f"Error creating routing run: {e}",
                    "specialists": [],
                    "reason": None,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return {"success": False,
                        "error": "Routing decision timed out",
                        "specialists": [],
                        "reason": None,
                        "user_query": user_message,
                        "raw_response": None
                        }
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                return {"success": False,
                        "error": f"Error checking routing status: {e}",
                        "specialists": [],
                        "reason": None,
                        "user_query": user_message,
                        "raw_response": None
                        }
        logging.info(f'ORCHESTRATOR RUN COMPLETED:\nInput tokens: {run.usage.prompt_tokens} ({run.usage.prompt_tokens*price_per_token_in}$)\nInput tokens: {run.usage.completion_tokens} ({run.usage.completion_tokens*price_per_token_out}$)')
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            routing_decision_raw = messages.data[0].content[0].text.value
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return {"success": False,
                    "error": f"Error retrieving routing decision: {e}",
                    "specialists": [],
                    "reason": None,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Parse decision - try to extract JSON from response
        try:
            # Look for JSON in the response
            OrchestratorResponse.model_validate_json(routing_decision_raw)
            orchestrator_response_dict = json.loads(routing_decision_raw)

            if len(orchestrator_response_dict['specialists']) > 0:   # TODO: Do we need this check if we have model validation???
                assistants_names = orchestrator_response_dict.get('specialists', None)
            else:
                # TODO: Fallback mechanism
                return {"success": False,
                        "error": "Empty assistants list",
                        "specialists": [],
                        "reason": None,
                        "user_query": user_message,
                        "raw_response": None
                        }
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Error parsing orchestrator response: {e}",
                "specialists": [],
                "reason": None,
                "user_query": user_message,
                "raw_response": None
                }
        
        # Log routing decision
        if session_id in self.context_store:
            self.context_store[session_id]['routing_decisions'].append(orchestrator_response_dict)
        
        return orchestrator_response_dict
    
    def get_bot_response(self, session_id: str, user_message: str, specialists_responses: list[dict]) -> dict:
        """Use combinator assistant to prepare the final answer"""
        specialists_names = [v['specialist'] for v in specialists_responses]

        if session_id not in self.shared_threads:
            return {"success": False,
                    "error": "Session not found",
                    "specialists": specialists_names,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Get current context
        # context = self.get_shared_context(session_id)
        
        # Create combinator prompt
        specialist_responses = ""
        for i, item in enumerate(specialists_responses):
            specialist_responses += f"SPECILAIST {i+1} DATA START\n"
            specialist_responses += f"SPECIALIST NAME: {item['specialist']}\n"
            specialist_responses += f"SPECIALIST RESPONSE: {item['response']}\n"
            specialist_responses += f"SPECILAIST {i+1} DATA END\n\n"

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
            return {"success": False,
                    "error": f"Error creating routing message: {e}",
                    "specialists": specialists_names,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
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
            return {"success": False,
                    "error": f"Error creating routing run: {e}",
                    "specialists": specialists_names,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return {"success": False,
                        "error": "Combinator call timed out",
                        "specialists": specialists_names,
                        "response": None,
                        "sources": [],
                        "images": [],
                        "user_query": user_message,
                        "raw_response": None
                        }
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                return {"success": False,
                        "error": f"Error checking combinator status: {e}",
                        "specialists": specialists_names,
                        "response": None,
                        "sources": [],
                        "images": [],
                        "user_query": user_message,
                        "raw_response": None
                        }
        
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            final_response = messages.data[0].content[0].text.value

            # Extract sources
            sources_list = []
            for spec_response in specialists_responses:
                sources_list += spec_response.get('sources')
                sources_list = list(set(sources_list))

            # Extract images
            img_list = []
            for spec_response in specialists_responses:
                img_list += spec_response.get('images')
                img_list = list(set(img_list))

        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return {"success": False,
                    "error": f"Error retrieving combinator response: {e}",
                    "specialists": specialists_names,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # # Parse decision - try to extract JSON from response
        # try:
        #     # Look for JSON in the response
        #     OrchestratorResponse.model_validate_json(routing_decision_raw)
        #     orchestrator_response_dict = json.loads(routing_decision_raw)

        #     if len(orchestrator_response_dict['specialists']) > 0:   # TODO: Do we need this check if we have model validation???
        #         assistants_names = orchestrator_response_dict.get('specialists', None)
        #     else:
        #         # TODO: Fallback mechanism
        #         return {"success": False,
        #                 "error": "Empty assistants list",
        #                 "specialists": [],
        #                 "reasoning": None,
        #                 "user_query": user_message,
        #                 "raw_response": None
        #                 }
        # except Exception as e:
        #     return {"success": False,
        #             "error": f"Error parsing orchestrator response: {e}",
        #             "specialists": [],
        #             "reasoning": None,
        #             "user_query": user_message,
        #             "raw_response": None
        #             }
        
        # Log routing decision
        # if session_id in self.context_store:
        #     self.context_store[session_id]['final_responses'].append(final_response)
        
        return {"success": True,
                "error": None,
                "specialists": specialists_names,
                "response": final_response,
                "sources": sources_list,
                "images": img_list,
                "user_query": user_message,
                "raw_response": final_response
                }
    
    
    def route_to_assistant(self, 
                           session_id: str, 
                           assistant_name: str, 
                           user_message: str, 
                           include_context: bool = True, 
                           new_thread: bool = False
                           ) -> dict:
        
        """Route message to specific assistant with context"""
        if session_id not in self.shared_threads:
            raise Exception(f"Session {session_id} not found")
        
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
            return {"success": False,
                    "error": f"Error creating run: {e}",
                    "specialist": assistant_name,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Wait for completion with timeout
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return {"success": False,
                        "error": "Assistant response timed out",
                        "specialist": assistant_name,
                        "response": None,
                        "sources": [],
                        "images": [],
                        "user_query": user_message,
                        "raw_response": None
                        }
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                # Clean up temp thread on error
                return {"success": False,
                        "error": f"Error retrieving run status: {e}",
                        "specialist": assistant_name,
                        "response": None,
                        "sources": [],
                        "images": [],
                        "user_query": user_message,
                        "raw_response": None
                        }
            
            if run.status == "failed":
                # Clean up temp thread on error
                return {"success": False,
                        "error": f"Assistant run failed: {run.last_error}",
                        "specialist": assistant_name,
                        "response": None,
                        "sources": [],
                        "images": [],
                        "user_query": user_message,
                        "raw_response": None
                        }
        
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
                files_path = assistant_files_mapping.get(assistant_name)
                try:
                    source_mapping_filepath = files_path + 'pdf_mapping.json'
                    with open(source_mapping_filepath, 'r', encoding='utf-8') as file:
                        pdf_mapping = json.load(file)
                    for source in sources_list:
                        source_filename = source.filename
                        name_without_ext = os.path.splitext(source_filename)[0]
                        source_file_path = pdf_mapping[name_without_ext]
                        sources_files_list.append(source_file_path)
                    sources_files_list = list(set(sources_files_list))
                except Exception as e:
                    logging.warning(f'Failed find source in pdf_mapping.json. {e}',exc_info=True)

            if len(img_markers_list) > 0:
                files_path = assistant_files_mapping.get(assistant_name)
                img_mapping_filepath = files_path + 'doc_mapping.json'
                with open(img_mapping_filepath, 'r', encoding='utf-8') as file:
                    img_mapping = json.load(file)
                for img in img_markers_list:
                    img_info = extract_marker_parts(marker = img)
                    if img_info:
                        img_dir = img_mapping[img_info['img_file_key']]
                        file = find_file_by_name(files_path+img_dir, img_info['img_file_key']+'_'+img_info['img_name'])  # TODO Kostyl
                        img_files_list.append(file[0].replace('\\', '/'))
                img_files_list = list(set(img_files_list))

            response_clean = delete_sources_from_text(text_wo_markers)

        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            # Clean up temp thread on error
            return {"success": False,
                    "error": f"Error processing response: {e}",
                    "specialist": assistant_name,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        response = {"success": True,
                    "error": None,
                    "specialist": assistant_name,
                    "response": response_clean,
                    "sources": sources_files_list,
                    "images": img_files_list,
                    "user_query": user_message,
                    "raw_response": messages.data[0].content[0].text.value
                    }
        
        # Update context
        if session_id in self.context_store:
            self.context_store[session_id]['last_assistant'] = assistant_name
            self.context_store[session_id]['conversation_history'].append({
                'timestamp': time.time(),
                'assistant': assistant_name,
                'role': 'assistant',
                'content': response,
                'message_type': 'response'
            })
        return response
    
    async def route_to_assistant_async(self, 
                                       session_id: str, 
                                       assistant_name: str, 
                                       user_message: str, 
                                       include_context: bool = True, 
                                       new_thread: bool = True) -> dict:
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
    
    async def process_user_request(self, session_id: str, user_message: str, telegram_user_id: Optional[int] = None) -> dict:
        """Complete request processing with routing and context"""
        
        # Ensure session exists
        if session_id not in self.context_store:
            session_id, session_msg = self.get_or_create_session(telegram_user_id)
        
        # Determine which assistant to use
        return self.process_with_orchestrator(session_id, user_message)
    
    def call_specialists_sequentially(self, session_id:str, specialists_names: list[str], user_message:str) -> dict:
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

        # TODO: Check how do we detect successfull messages
        for i, response in enumerate(specialist_responses):
            if isinstance(response, Exception):
                failed_responses.append(response)
            else:
                successful_responses.append(response)

        return {"successful_responses": successful_responses, "failed_responses": failed_responses}


    async def call_specialists_parallel(self, session_id: str, specialists_names: list[str], user_message:str) -> dict:
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

            # TODO: Check how do we detect successfull messages
            for i, response in enumerate(specialist_responses):
                if isinstance(response, Exception):
                    failed_responses.append(response)
                else:
                    successful_responses.append(response)

            return {"successful_responses": successful_responses, "failed_responses": failed_responses}


        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            return {"successful_responses": [], "failed_responses": []}

# Convenience functions for easy integration
def create_orchestrator(api_key: str = None) -> TelegramMultiAssistantOrchestrator:
    """Create and return configured orchestrator"""
    if api_key is None:
        api_key = os.environ.get("OPENAI_TOKEN")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_TOKEN environment variable not set")
    
    return TelegramMultiAssistantOrchestrator(api_key)
