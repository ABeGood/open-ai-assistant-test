from openai import OpenAI
import json
import time
import os
import uuid
import re
import asyncio
from typing import Dict, Optional, Tuple
from validators import OrchestratorResponse
from response_processing_utils import (
    get_all_markers_as_list, 
    remove_all_markers, 
    delete_sources_from_text, 
    assistant_files_mapping, 
    extract_marker_parts,
    find_file_by_name
)


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
        assistant_configs = {
            "orchestrator": {
                "id": "asst_aU6DIODwxNlFRxrY3WipBPjz",
                "purpose": "Route requests and coordinate other assistants"
            },
            "equipment": {
                "id": "asst_1dQLsAz9p6T2cQyGtnjSeXnv",
                "purpose": "Equipment expert"
            },
            "tools": {
                "id": "asst_jtOdIxiHK1UsVkXaCxM8y0PS",
                "purpose": "Tools expert"
            },
            "cables": {
                "id": "asst_cErO4m6RZdfHQPAT3wVagp2z",
                "purpose": "Cables for equipment connection expert"
            },
            "commonInfo": {
                "id": "asst_nJzNpbdII7UzbOGiiSFcu09u",
                "purpose": "Expert with common information knowledge"
            },
            "combinator": {
                "id": "asst_FM5jrNCeRHxy3MpMueV1RkED",
                "purpose": "Combine experts responses into a final response for user"
            }
        }
        
        for name, config in assistant_configs.items():
            self.register_assistant(name, config["id"], config["purpose"])
    
    def register_assistant(self, name: str, assistant_id: str, purpose: str):
        """Register an assistant in the orchestrator"""
        try:
            self.assistants[name] = {
                'id': assistant_id,
                'purpose': purpose,
                'assistant_obj': self.client.beta.assistants.retrieve(assistant_id)
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
    
    def get_or_create_session(self, telegram_user_id: Optional[int] = None) -> Tuple[str, str]:
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
            return {"success": False,
                    "error": "Session not found",
                    "specialists": [],
                    "reasoning": None,
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
        
        # Add routing message
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=routing_prompt,
                metadata={"type": "routing_decision"}
            )
        except Exception as e:
            return {"success": False,
                    "error": f"Error creating routing message: {e}",
                    "specialists": [],
                    "reasoning": None,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Get routing decision
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=orchestrator_id
            )
        except Exception as e:
            return {"success": False,
                    "error": f"Error creating routing run: {e}",
                    "specialists": [],
                    "reasoning": None,
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
                        "reasoning": None,
                        "user_query": user_message,
                        "raw_response": None
                        }
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                return {"success": False,
                        "error": f"Error checking routing status: {e}",
                        "specialists": [],
                        "reasoning": None,
                        "user_query": user_message,
                        "raw_response": None
                        }
        
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            routing_decision_raw = messages.data[0].content[0].text.value
        except Exception as e:
            return {"success": False,
                    "error": f"Error retrieving routing decision: {e}",
                    "specialists": [],
                    "reasoning": None,
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
                        "reasoning": None,
                        "user_query": user_message,
                        "raw_response": None
                        }
        except Exception as e:
            return {"success": False,
                    "error": f"Error parsing orchestrator response: {e}",
                    "specialists": [],
                    "reasoning": None,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Log routing decision
        if session_id in self.context_store:
            self.context_store[session_id]['routing_decisions'].append(orchestrator_response_dict)
        
        return orchestrator_response_dict
    
    def process_with_combinator(self, session_id: str, user_message: str, specialists_responses: list[dict]) -> dict:
        """Use combinator assistant to prepare the final answer"""
        specialists_names = [v['specialist'] for v in specialists_responses]

        if session_id not in self.shared_threads:
            return {"success": False,
                    "error": "Session not found",
                    "specialists": specialists_names,
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
            specialist_responses += f"SPECIALIST RESPONSE: {item['response']['response']}\n"
            specialist_responses += f"SPECILAIST {i+1} DATA END\n\n"

        combinator_prompt = f"""USER QUERY: 
{user_message}

{specialist_responses}
"""
        
        # Route to combinator for decision
        thread_id = self.shared_threads[session_id]
        combinator_id = self.assistants['combinator']['id']
        
        # Add routing message
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=combinator_prompt,
                metadata={"type": "combinator_call"}
            )
        except Exception as e:
            return {"success": False,
                    "error": f"Error creating routing message: {e}",
                    "specialists": specialists_names,
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Get final answer
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=combinator_id
            )
        except Exception as e:
            return {"success": False,
                    "error": f"Error creating routing run: {e}",
                    "specialists": specialists_names,
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
                        "user_query": user_message,
                        "raw_response": None
                        }
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                return {"success": False,
                        "error": f"Error checking combinator status: {e}",
                        "specialists": specialists_names,
                        "user_query": user_message,
                        "raw_response": None
                        }
        
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            final_response = messages.data[0].content[0].text.value
        except Exception as e:
            return {"success": False,
                    "error": f"Error retrieving combinator response: {e}",
                    "specialists": specialists_names,
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
        if session_id in self.context_store:
            self.context_store[session_id]['final_response'].append(final_response)
        
        return {"success": True,
                    "error": None,
                    "specialists": specialists_names,
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

        # Determine which thread to use
        if new_thread:
            # Force creation of temporary thread for parallel execution
            try:
                temp_thread = self.client.beta.threads.create()
                thread_id = temp_thread.id
                use_temp_thread = True
            except Exception as e:
                return {"success": False,
                        "error": f"Error creating temporary thread: {e}",
                        "specialist": assistant_name,
                        "response": None,
                        "sources": [],
                        "images": [],
                        "user_query": user_message,
                        "raw_response": None
                        }
        else:
            # Use shared thread, but check if it's busy first
            thread_id = self.shared_threads[session_id]

            try:
                runs = self.client.beta.threads.runs.list(thread_id=thread_id, limit=5)
                for existing_run in runs.data:
                    if existing_run.status in ['queued', 'in_progress', 'requires_action']:
                        # Thread is busy, create a temporary thread
                        try:
                            temp_thread = self.client.beta.threads.create()
                            thread_id = temp_thread.id
                            use_temp_thread = True
                            break
                        except Exception as e:
                            return {"success": False,
                                    "error": f"Error creating temporary thread: {e}",
                                    "specialist": assistant_name,
                                    "response": None,
                                    "sources": [],
                                    "images": [],
                                    "user_query": user_message,
                                    "raw_response": None
                                    }
            except Exception as e:
                # If we can't check runs, proceed with original thread
                pass
            
        # Add context if requested
        if include_context and session_id in self.context_store:
            # context = self.get_shared_context(session_id)
            full_message = f"USER REQUEST: {user_message}"
        else:
            full_message = user_message
        
        # Add user message to thread
        try:
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=full_message,
                metadata={"routed_to": assistant_name, "temp_thread": str(use_temp_thread)}
            )
        except Exception as e:
            if use_temp_thread:
                try:
                    self.client.beta.threads.delete(thread_id=thread_id)
                except:
                    pass
            return {"success": False,
                    "error": f"Error creating message: {e}",
                    "specialist": assistant_name,
                    "response": None,
                    "sources": [],
                    "images": [],
                    "user_query": user_message,
                    "raw_response": None
                    }
        
        # Run with specific assistant
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
        except Exception as e:
            # Clean up temp thread if run creation failed
            if use_temp_thread:
                try:
                    self.client.beta.threads.delete(thread_id=thread_id)
                except:
                    pass
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
                # Clean up temp thread on error
                if use_temp_thread:
                    try:
                        self.client.beta.threads.delete(thread_id=thread_id)
                    except:
                        pass
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
                if use_temp_thread:
                    try:
                        self.client.beta.threads.delete(thread_id=thread_id)
                    except:
                        pass
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

            img_markers_list = get_all_markers_as_list(messages.data[0].content[0].text.value)

            sources_files_list = []
            img_files_list = []

            if len(sources_list) > 0:
                files_path = assistant_files_mapping.get(assistant_name)
                source_mapping_filepath = files_path + 'pdf_mapping.json'
                with open(source_mapping_filepath, 'r', encoding='utf-8') as file:
                    pdf_mapping = json.load(file)
                for source in sources_list:
                    source_filename = source.filename
                    name_without_ext = os.path.splitext(source_filename)[0]
                    source_file_path = pdf_mapping[name_without_ext]
                    sources_files_list.append(source_file_path)

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

            response_clean = delete_sources_from_text(messages.data[0].content[0].text.value)
            response_clean = remove_all_markers(response_clean)

        except Exception as e:
            # Clean up temp thread on error
            if use_temp_thread:
                try:
                    self.client.beta.threads.delete(thread_id=thread_id)
                except:
                    pass
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

        # Clean up temp thread on error
        if use_temp_thread:
            try:
                self.client.beta.threads.delete(thread_id=thread_id)
            except:
                pass
        
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
    
    async def process_request(self, session_id: str, user_message: str, telegram_user_id: Optional[int] = None) -> Dict:
        """Complete request processing with routing and context"""
        
        # Ensure session exists
        if session_id not in self.context_store:
            session_id, session_msg = self.get_or_create_session(telegram_user_id)
        
        # Determine which assistant to use
        orchestrator_response_dict = self.process_with_orchestrator(session_id, user_message)
        
        chosen_specialists = orchestrator_response_dict.get('specialists', None)

        if not chosen_specialists:
            return {"success": False,
                    "error": f"Empty specialists list",
                    "user_query": user_message,
                    "raw_response": None
                    }
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
        tasks = [call_single_assistant(specialist) for specialist in chosen_specialists]

        try:
            specialist_responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process responses
            successful_responses = []
            failed_responses = []

            # TODO: Check how do we detect successfull messages
            for i, response in enumerate(specialist_responses):
                if isinstance(response, Exception):
                    failed_responses.append({
                        'specialist': chosen_specialists[i],
                        'error': str(response)
                    })
                else:
                    successful_responses.append({
                        'specialist': chosen_specialists[i],
                        'response': response
                    })

            final_answer = self.process_with_combinator(session_id, user_message, successful_responses)

            
            
            return {
                'assistants_used': chosen_specialists,
                'successful_responses': successful_responses,
                'failed_responses': failed_responses,
                'session_id': session_id,
                'routing_decision': orchestrator_response_dict,
                'timestamp': time.time()
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Parallel execution failed: {str(e)}",
                "user_query": user_message,
                "specialists_attempted": chosen_specialists
            }
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        if session_id not in self.context_store:
            return {"error": "Session not found"}
        
        context = self.context_store[session_id]
        return {
            "session_id": session_id,
            "telegram_user_id": context.get('telegram_user_id'),
            "created_at": context.get('created_at'),
            "conversation_count": len([h for h in context['conversation_history'] if h['role'] == 'user']),
            "last_assistant": context.get('last_assistant'),
            "routing_decisions_count": len(context.get('routing_decisions', []))
        }
    
    def list_sessions(self) -> Dict:
        """List all active sessions"""
        sessions = {}
        for session_id in self.context_store:
            sessions[session_id] = self.get_session_info(session_id)
        return sessions


# Convenience functions for easy integration
def create_orchestrator(api_key: str = None) -> TelegramMultiAssistantOrchestrator:
    """Create and return configured orchestrator"""
    if api_key is None:
        api_key = os.environ.get("OPENAI_TOKEN")
    
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_TOKEN environment variable not set")
    
    return TelegramMultiAssistantOrchestrator(api_key)


async def process_telegram_message(orchestrator: TelegramMultiAssistantOrchestrator, 
                           user_message: str, 
                           telegram_user_id: int) -> Dict:
    """
    Process a message from Telegram user
    
    Args:
        orchestrator: The orchestrator instance
        user_message: The user's message
        telegram_user_id: Telegram user ID
    
    Returns:
        Dict with response data including routing decisions for debugging
    """
    session_id = f"tg-{telegram_user_id}"
    return await orchestrator.process_request(session_id, user_message, telegram_user_id)


# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize orchestrator
#     orchestrator = create_orchestrator()
    
#     # Test with telegram user
#     telegram_user_id = 123456789
    
#     # Process some messages
#     test_messages = [
#         "что такое MS50183-EPS?",
#         "какие кабели нужны для диагностики генераторов?",
#         "сбросить контекст"
#     ]
    
#     print("=== Testing Telegram Multi-Assistant Orchestrator ===")
    
#     for i, message in enumerate(test_messages):
#         print(f"\n--- Message {i+1}: {message} ---")
        
#         if message.lower() in ["сбросить контекст", "reset context", "/reset"]:
#             session_id = f"tg-{telegram_user_id}"
#             result = orchestrator.reset_context(session_id)
#             print(f"Context reset: {result}")
#             continue
        
#         result = process_telegram_message(orchestrator, message, telegram_user_id)
        
#         print(f"Assistant: {result['assistant_used']}")
#         print(f"Routing: {result['routing_decision']}")
#         print(f"Response: {result['response'][:200]}...")
    
#     # Show session info
#     session_id = f"tg-{telegram_user_id}"
#     session_info = orchestrator.get_session_info(session_id)
#     print(f"\n=== Session Info ===")
#     print(json.dumps(session_info, indent=2))