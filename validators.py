from pydantic import BaseModel, Field
from enum import Enum

class SpecialistType(str, Enum):
    """Enum for valid specialist types"""
    EQUIPMENT = "equipment"
    DIAGNOSTICS = "diagnostics"
    COMPATIBILITY = "compatibility"
    TOOLS = "tools"
    CABLES = "cables"
    SUPPORT = "support"


class OrchestratorResponse(BaseModel):
    """Pydantic model for specialist routing response validation"""
    
    user_query: str = Field(
        ..., 
        description="The original user question or query"
    )
    
    specialists: list[SpecialistType] = Field(
        ..., 
        min_length=1,
        description="List of specialist types assigned to handle the query"
    )
    
    reason: str = Field(
        ..., 
        description="Explanation of why these specific specialists were chosen for the query"
    )
    
    class Config:
        extra = 'forbid'  # Equivalent to additionalProperties: false
        use_enum_values = True  # Use enum values in serialization