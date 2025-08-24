from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from classes.classes import Message
from classes.validators import SpecialistRoutingResponse
from agents.orchestrator.prompts.prompts import ORCHESTRATOR_PROMPT


class OrchestratorAgent:
    """Orchestrator agent using chat completions API for routing queries to specialists"""
    
    def __init__(self, llm_client: OpenAI, llm_model: str = "gpt-4o-2024-08-06"):
        self.llm_client = llm_client
        self.llm_model = llm_model
    
    def route_query_chat_completion(
        self, 
        user_query: str, 
        last_n_messages: List[Message],
        user_message_metadata: str = ""
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
        # Format conversation history
        conversation_history = self._format_conversation_history(last_n_messages)
        
        # Create prompt from template
        prompt = ORCHESTRATOR_PROMPT.format(
            conversation_history=conversation_history,
            user_message=user_query,
            user_message_metadata=user_message_metadata
        )
        
        # Get JSON schema from Pydantic model
        schema = {
            "name": "specialist_routing_response",
            "strict": True,
            "schema": SpecialistRoutingResponse.model_json_schema()
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
            formatted_messages.append(f"{author}: {content}")
        
        return "\n".join(formatted_messages)