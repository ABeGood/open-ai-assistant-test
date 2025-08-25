from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from classes.classes import Message
from classes.validators import SpecialistRoutingResponse, flatten_schema
from agents.orchestrator.prompts.prompts import ORCHESTRATOR_PROMPT
from agents.prompt_static_analyzer.prompt_static_analyzer import PromptStaticAnalyzer


class OrchestratorAgent:
    """Orchestrator agent using chat completions API for routing queries to specialists"""
    
    def __init__(self, llm_client: OpenAI, llm_model: str = "gpt-4o-2024-08-06"):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.prompt_static_analyzer = PromptStaticAnalyzer()
    
    def route_query_chat_completion(
        self, 
        user_query: str, 
        last_n_messages: List[Message],
    ) -> SpecialistRoutingResponse:
        """
        Route user query to appropriate specialists using chat completions API
        
        Args:
            user_query: The user's current query
            last_n_messages: Recent conversation messages for context
            user_message_metadata: Additional metadata from static analysis
            
        Returns:
            SpecialistRoutingResponse: Validated routing decision
        """
        user_message_metadata = self.prompt_static_analyzer.route_query(user_query)

        # Format conversation history
        conversation_history = self._format_conversation_history(last_n_messages[-6:-1])
        
        # Create prompt from template
        prompt = ORCHESTRATOR_PROMPT.format(
            conversation_history=conversation_history,
            user_message=user_query,
            user_message_metadata=user_message_metadata 
            # user_message_metadata= 'No message metadata'
        )
        
        # Get JSON schema from Pydantic model
        raw_schema = SpecialistRoutingResponse.model_json_schema()
        clean_schema = flatten_schema(raw_schema)

        # Ensure required includes all properties
        schema = {
            "name": "specialist_routing_response",
            "strict": True,
            "schema": clean_schema
        }
        
        # Call OpenAI with structured output
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": prompt}
        ]
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": schema
            },
            temperature=0.01
        )
        
        # Parse and validate response
        response_content = response.choices[0].message.content
        routing_result = SpecialistRoutingResponse.model_validate_json(response_content)
        
        return routing_result
    
    def _format_conversation_history(self, messages: List[Message]) -> str:
        """Format conversation messages into readable history string"""
        if not messages:
            return "No previous conversation history."
        
        formatted_messages = []
        for msg in messages:
            author = msg.get_author()
            content = msg.get_content()
            author = 'Assistant' if str(author).startswith('bot') else 'User'
            formatted_messages.append(f"{author}: {content}")
        
        return "\n".join(formatted_messages)