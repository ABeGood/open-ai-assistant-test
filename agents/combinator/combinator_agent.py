from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import time
import logging

from classes.agents_response_models import (
    SpecialistResponse,
    CombinatorResponse,
    create_success_response,
    create_error_response
)
from agents.combinator.prompts.prompts import COMBINATOR_PROMPT


class CombinatorAgent:
    """Combinator agent using chat completions API for combining specialist responses"""
    
    def __init__(self, llm_client: OpenAI, llm_model: str = "gpt-4o-2024-08-06"):
        self.llm_client = llm_client
        self.llm_model = llm_model
    
    def process_with_combinator_chat(
        self, 
        user_message: str, 
        specialists_responses: List[SpecialistResponse]
    ) -> CombinatorResponse:
        """
        Use combinator with chat completions API to prepare the final answer
        
        Args:
            user_message: The original user query
            specialists_responses: List of specialist responses to combine
            
        Returns:
            CombinatorResponse: Combined response from all specialists
        """
        specialists_names = [resp.specialist for resp in specialists_responses]
        
        # Create specialist responses text
        specialist_responses_text = ""
        for i, item in enumerate(specialists_responses):
            specialist_responses_text += f"SPECIALIST {i+1} DATA START\n"
            specialist_responses_text += f"SPECIALIST NAME: {item.specialist}\n"
            specialist_responses_text += f"SPECIALIST RESPONSE: {item.response}\n"
            specialist_responses_text += f"SPECIALIST {i+1} DATA END\n\n"

        # Create the final prompt using the imported template
        full_prompt = COMBINATOR_PROMPT.format(
            user_message=user_message,
            specialist_responses=specialist_responses_text
        )
        
        # Call OpenAI chat completions
        try:
            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": full_prompt}
            ]
            
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            final_response = response.choices[0].message.content
            
            if not final_response:
                return create_error_response(
                    CombinatorResponse,
                    "Empty response from combinator",
                    user_message,
                    specialists=specialists_names
                )
            
        except Exception as e:
            logging.error(f"Error calling combinator chat completion: {e}", exc_info=True)
            return create_error_response(
                CombinatorResponse,
                f"Error calling combinator: {e}",
                user_message,
                specialists=specialists_names
            )
        
        # Extract sources and images from all specialist responses
        sources_list = []
        img_list = []
        
        for spec_response in specialists_responses:
            sources_list.extend(spec_response.sources)
            img_list.extend(spec_response.images)
        
        # Remove duplicates
        sources_list = list(set(sources_list))
        img_list = list(set(img_list))
        
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
