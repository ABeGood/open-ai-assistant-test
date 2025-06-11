from openai import OpenAI
import json
import time
import os
import uuid
import re
from typing import Dict, Optional, Tuple


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
    
    def create_session(self, telegram_user_id: Optional[int] = None) -> Tuple[str, str]:
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
    
    def add_context_message(self, session_id: str, assistant_name: str, content: str, role: str = "assistant"):
        """Add contextual message to maintain consistency"""
        if session_id not in self.context_store:
            return
        
        thread_id = self.shared_threads[session_id]
        
        # Add to context store
        context_entry = {
            'timestamp': time.time(),
            'assistant': assistant_name,
            'role': role,
            'content': content,
            'message_type': 'context_update'
        }
        
        self.context_store[session_id]['conversation_history'].append(context_entry)
        
        # Add hidden context message to thread
        context_message = f"[CONTEXT UPDATE from {assistant_name}]: {content}"
        
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="assistant",
                content=context_message,
                metadata={"type": "context", "source_assistant": assistant_name}
            )
        except Exception as e:
            print(f"Error adding context message: {e}")
    
    def get_shared_context(self, session_id: str) -> str:
        """Get formatted shared context for assistant"""
        if session_id not in self.context_store:
            return "=== NO CONTEXT ==="
        
        context = self.context_store[session_id]
        
        context_summary = "=== SHARED CONTEXT ===\n"
        context_summary += f"Previous assistants involved: {set(h['assistant'] for h in context['conversation_history'])}\n"
        context_summary += f"Last assistant: {context['last_assistant']}\n"
        
        # Recent conversation summary
        recent_history = context['conversation_history'][-5:]  # Last 5 interactions
        context_summary += "\nRECENT INTERACTIONS:\n"
        
        for entry in recent_history:
            if entry['role'] == 'user':
                context_summary += f"User: {entry['content'][:100]}...\n"
            else:
                context_summary += f"{entry['assistant']}: {entry['content'][:100]}...\n"
        
        # Shared data
        if context['shared_context']:
            context_summary += f"\nSHARED DATA: {json.dumps(context['shared_context'], indent=2)}\n"
        
        context_summary += "=== END CONTEXT ===\n"
        
        return context_summary
    
    def route_to_assistant(self, session_id: str, assistant_name: str, user_message: str, include_context: bool = True) -> str:
        """Route message to specific assistant with context"""
        if session_id not in self.shared_threads:
            raise Exception(f"Session {session_id} not found")
        
        thread_id = self.shared_threads[session_id]
        assistant_id = self.assistants[assistant_name]['id']
        
        # Add context if requested
        if include_context and session_id in self.context_store:
            context = self.get_shared_context(session_id)
            full_message = f"{context}\n\nUSER REQUEST: {user_message}"
        else:
            full_message = user_message
        
        # Add user message to thread
        try:
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=full_message,
                metadata={"routed_to": assistant_name}
            )
        except Exception as e:
            return f"Error creating message: {e}"
        
        # Run with specific assistant
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
        except Exception as e:
            return f"Error creating run: {e}"
        
        # Wait for completion with timeout
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return "Assistant response timed out"
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                return f"Error retrieving run status: {e}"
            
            if run.status == "failed":
                return f"Assistant run failed: {run.last_error}"
        
        # Get response
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            response = messages.data[0].content[0].text.value
        except Exception as e:
            return f"Error retrieving response: {e}"
        
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
    
    def analyze_request(self, session_id: str, user_message: str) -> Tuple[str, Dict]:
        """Use orchestrator assistant to determine routing"""
        if session_id not in self.shared_threads:
            return "orchestrator", {"error": "Session not found"}
        
        # Get current context
        context = self.get_shared_context(session_id)
        
        # Create routing prompt
        routing_prompt = f"""
{context}

USER REQUEST: {user_message}
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
            return "orchestrator", {"error": f"Error creating routing message: {e}"}
        
        # Get routing decision
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=orchestrator_id
            )
        except Exception as e:
            return "orchestrator", {"error": f"Error creating routing run: {e}"}
        
        # Wait for completion
        timeout = 30
        start_time = time.time()
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                return "orchestrator", {"error": "Routing decision timed out"}
            
            time.sleep(1)
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            except Exception as e:
                return "orchestrator", {"error": f"Error checking routing status: {e}"}
        
        try:
            messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            routing_decision_raw = messages.data[0].content[0].text.value
        except Exception as e:
            return "orchestrator", {"error": f"Error retrieving routing decision: {e}"}
        
        # Parse decision - try to extract JSON from response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]*"specialist"[^}]*\}', routing_decision_raw)
            if json_match:
                routing_decision_dict = json.loads(json_match.group())
                assistant_name = routing_decision_dict.get('specialist', 'orchestrator')
            else:
                # Fallback: look for assistant names in text
                assistant_name = "orchestrator"
                for name in self.assistants.keys():
                    if name.lower() in routing_decision_raw.lower():
                        assistant_name = name
                        break
                
                routing_decision_dict = {
                    "specialist": assistant_name,
                    "reasoning": routing_decision_raw,
                    "fallback_parsing": True
                }
        except Exception as e:
            # Ultimate fallback
            assistant_name = "orchestrator"
            routing_decision_dict = {
                "specialist": assistant_name,
                "reasoning": routing_decision_raw,
                "error": f"JSON parsing failed: {e}",
                "raw_response": routing_decision_raw
            }
        
        # Validate assistant exists
        if assistant_name not in self.assistants:
            assistant_name = 'orchestrator'  # Fallback
            routing_decision_dict["fallback_to_orchestrator"] = True
        
        # Log routing decision
        if session_id in self.context_store:
            self.context_store[session_id]['routing_decisions'].append({
                'timestamp': time.time(),
                'user_message': user_message,
                'decision': routing_decision_dict,
                'chosen_assistant': assistant_name
            })
        
        return assistant_name, routing_decision_dict
    
    def process_request(self, session_id: str, user_message: str, telegram_user_id: Optional[int] = None) -> Dict:
        """Complete request processing with routing and context"""
        
        # Ensure session exists
        if session_id not in self.context_store:
            session_id, session_msg = self.get_or_create_session(telegram_user_id)
        
        # Determine which assistant to use
        chosen_assistant, routing_decision = self.analyze_request(session_id, user_message)
        
        # Route to chosen assistant
        response = self.route_to_assistant(
            session_id=session_id,
            assistant_name=chosen_assistant,
            user_message=user_message,
            include_context=True
        )
        
        return {
            'assistant_used': chosen_assistant,
            'response': response,
            'session_id': session_id,
            'routing_decision': routing_decision,
            'timestamp': time.time()
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


def process_telegram_message(orchestrator: TelegramMultiAssistantOrchestrator, 
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
    return orchestrator.process_request(session_id, user_message, telegram_user_id)


# Example usage and testing
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = create_orchestrator()
    
    # Test with telegram user
    telegram_user_id = 123456789
    
    # Process some messages
    test_messages = [
        "что такое MS50183-EPS?",
        "какие кабели нужны для диагностики генераторов?",
        "сбросить контекст"
    ]
    
    print("=== Testing Telegram Multi-Assistant Orchestrator ===")
    
    for i, message in enumerate(test_messages):
        print(f"\n--- Message {i+1}: {message} ---")
        
        if message.lower() in ["сбросить контекст", "reset context", "/reset"]:
            session_id = f"tg-{telegram_user_id}"
            result = orchestrator.reset_context(session_id)
            print(f"Context reset: {result}")
            continue
        
        result = process_telegram_message(orchestrator, message, telegram_user_id)
        
        print(f"Assistant: {result['assistant_used']}")
        print(f"Routing: {result['routing_decision']}")
        print(f"Response: {result['response'][:200]}...")
    
    # Show session info
    session_id = f"tg-{telegram_user_id}"
    session_info = orchestrator.get_session_info(session_id)
    print(f"\n=== Session Info ===")
    print(json.dumps(session_info, indent=2))