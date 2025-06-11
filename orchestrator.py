from openai import OpenAI
import json
import time
import os

class MultiAssistantOrchestrator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.assistants = {}
        self.shared_threads = {}
        self.context_store = {}
        
    def register_assistant(self, name: str, assistant_id: str, purpose: str):
        """Register an assistant in the orchestrator"""
        self.assistants[name] = {
            'id': assistant_id,
            'purpose': purpose,
            'assistant_obj': self.client.beta.assistants.retrieve(assistant_id)
        }
        
    def create_shared_thread(self, session_id: str) -> str:
        """Create a shared thread for the session"""
        thread = self.client.beta.threads.create()
        self.shared_threads[session_id] = thread.id
        self.context_store[session_id] = {
            'conversation_history': [],
            'shared_context': {},
            'last_assistant': None,
            'routing_decisions': []
        }
        return thread.id

# Initialize orchestrator
orchestrator = MultiAssistantOrchestrator(os.environ.get("OPENAI_TOKEN"))

# Register specialized assistants
orchestrator.register_assistant(
    name="orchestrator", 
    assistant_id="asst_aU6DIODwxNlFRxrY3WipBPjz", 
    purpose="Route requests and coordinate other assistants"
)

orchestrator.register_assistant(
    name="equipment", 
    assistant_id="asst_1dQLsAz9p6T2cQyGtnjSeXnv", 
    purpose="Equipment expert"
)

orchestrator.register_assistant(
    name="tools", 
    assistant_id="asst_jtOdIxiHK1UsVkXaCxM8y0PS", 
    purpose="Tools expert"
)

orchestrator.register_assistant(
    name="cables", 
    assistant_id="asst_cErO4m6RZdfHQPAT3wVagp2z", 
    purpose="Cables for equipment connection expert"
)

orchestrator.register_assistant(
    name="commonInfo", 
    assistant_id="asst_nJzNpbdII7UzbOGiiSFcu09u", 
    purpose="Expert with common information knowledge"
)


class ThreadConsistencyManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.client = orchestrator.client
    
    def add_context_message(self, session_id: str, assistant_name: str, content: str, role: str = "assistant"):
        """Add contextual message to maintain consistency"""
        
        thread_id = self.orchestrator.shared_threads[session_id]
        
        # Add to context store
        context_entry = {
            'timestamp': time.time(),
            'assistant': assistant_name,
            'role': role,
            'content': content,
            'message_type': 'context_update'
        }
        
        self.orchestrator.context_store[session_id]['conversation_history'].append(context_entry)
        
        # Add hidden context message to thread (not visible to user)
        context_message = f"[CONTEXT UPDATE from {assistant_name}]: {content}"
        
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=context_message,
            metadata={"type": "context", "source_assistant": assistant_name}
        )
    
    def get_shared_context(self, session_id: str) -> str:
        """Get formatted shared context for assistant"""
        
        context = self.orchestrator.context_store[session_id]
        
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
    
    def route_to_assistant(self, session_id: str, assistant_name: str, user_message: str, include_context: bool = True):
        """Route message to specific assistant with context"""
        
        thread_id = self.orchestrator.shared_threads[session_id]
        assistant_id = self.orchestrator.assistants[assistant_name]['id']
        
        # Add context if requested
        if include_context:
            context = self.get_shared_context(session_id)
            full_message = f"{context}\n\nUSER REQUEST: {user_message}"
        else:
            full_message = user_message
        
        # Add user message to thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=full_message,
            metadata={"routed_to": assistant_name}
        )
        
        # Run with specific assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        # Wait for completion
        while run.status != "completed":
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            
            if run.status == "failed":
                raise Exception(f"Assistant run failed: {run.last_error}")
        
        # Get response
        messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
        response = messages.data[0].content[0].text.value
        
        # Update context
        self.orchestrator.context_store[session_id]['last_assistant'] = assistant_name
        self.orchestrator.context_store[session_id]['conversation_history'].append({
            'timestamp': time.time(),
            'assistant': assistant_name,
            'role': 'assistant',
            'content': response,
            'message_type': 'response'
        })
        
        return response

# Initialize consistency manager
consistency_manager = ThreadConsistencyManager(orchestrator)

class IntelligentRouter:
    def __init__(self, orchestrator, consistency_manager):
        self.orchestrator = orchestrator
        self.consistency_manager = consistency_manager
        self.client = orchestrator.client
    
    def analyze_request(self, session_id: str, user_message: str) -> str:
        """Use orchestrator assistant to determine routing"""
        
        # Get current context
        context = self.consistency_manager.get_shared_context(session_id)
        
        # Create routing prompt
        routing_prompt = f"""
{context}

USER REQUEST: {user_message}
"""
        
        # Route to orchestrator for decision
        thread_id = self.orchestrator.shared_threads[session_id]
        orchestrator_id = self.orchestrator.assistants['orchestrator']['id']
        
        # Add routing message
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=routing_prompt,
            metadata={"type": "routing_decision"}
        )
        
        # Get routing decision
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=orchestrator_id
        )
        
        while run.status != "completed":
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        
        messages = self.client.beta.threads.messages.list(thread_id=thread_id, limit=1)
        routing_decision = messages.data[0].content[0].text.value
        routing_decision_dict = json.loads(routing_decision)
        # Parse decision
        assistant_name = routing_decision_dict.get('specialist', "NOT DEFINED")
        
        # Validate assistant exists
        if assistant_name not in self.orchestrator.assistants:
            assistant_name = 'orchestrator'  # Fallback
        
        # Log routing decision
        self.orchestrator.context_store[session_id]['routing_decisions'].append({
            'timestamp': time.time(),
            'user_message': user_message,
            'decision': routing_decision,
            'chosen_assistant': assistant_name
        })
        
        return assistant_name
    
    def process_request(self, session_id: str, user_message: str):
        """Complete request processing with routing and context"""
        
        # Determine which assistant to use
        chosen_assistant = self.analyze_request(session_id, user_message)
        
        print(f"🤖 Routing to: {chosen_assistant}")
        
        # Route to chosen assistant
        response = self.consistency_manager.route_to_assistant(
            session_id=session_id,
            assistant_name=chosen_assistant,
            user_message=user_message,
            include_context=True
        )
        
        return {
            'assistant_used': chosen_assistant,
            'response': response,
            'session_id': session_id
        }

# Initialize router
router = IntelligentRouter(orchestrator, consistency_manager)

def handoff_between_assistants(session_id: str, from_assistant: str, to_assistant: str, handoff_data: dict):
    """Seamless handoff between assistants with context preservation"""
    
    # Add handoff context
    handoff_message = f"""
=== HANDOFF FROM {from_assistant.upper()} TO {to_assistant.upper()} ===
Previous assistant: {from_assistant}
Handoff data: {json.dumps(handoff_data, indent=2)}

Continue the conversation seamlessly based on this context.
===
"""
    
    consistency_manager.add_context_message(
        session_id=session_id,
        assistant_name=from_assistant,
        content=handoff_message,
        role="assistant"
    )
    
    # Update shared context
    orchestrator.context_store[session_id]['shared_context'].update(handoff_data)
    
    return f"Handed off from {from_assistant} to {to_assistant}"

def share_data_between_assistants(session_id: str, data_key: str, data_value):
    """Share specific data between all assistants in session"""
    
    orchestrator.context_store[session_id]['shared_context'][data_key] = data_value
    
    # Notify all assistants about the shared data
    context_update = f"SHARED DATA UPDATED: {data_key} = {data_value}"
    
    consistency_manager.add_context_message(
        session_id=session_id,
        assistant_name="system",
        content=context_update,
        role="assistant"
    )

def run_multi_assistant_session(session_id:str):
    """Complete example of multi-assistant orchestration"""
    
    # Create session
    session_id = session_id
    thread_id = orchestrator.create_shared_thread(session_id)
    
    print(f"Created session {session_id} with thread {thread_id}")
    
    # Process multiple requests
    requests = [
        # "что такое MS015?",
        "что такое MS50183-EPS?",
    ]
    
    for i, request in enumerate(requests):
        print(f"\n=== REQUEST {i+1}: {request} ===")
        
        result = router.process_request(session_id, request)
        
        print(f"Assistant used: {result['assistant_used']}")
        print(f"Response: {result['response'][:200]}...")
        
        # Share results between assistants
        share_data_between_assistants(
            session_id=session_id,
            data_key=f"request_{i+1}_result",
            data_value=result['response'][:500]
        )
    
    # Final context summary
    final_context = consistency_manager.get_shared_context(session_id)
    print(f"\n=== FINAL SESSION CONTEXT ===")
    print(final_context)

# Run the session
run_multi_assistant_session()