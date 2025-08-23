from pydantic import BaseModel, Field
from enum import Enum

class SpecialistType(str, Enum):
    """Enum for available specialist types"""
    EQUIPMENT = "equipment"
    DIAGNOSTICS = "diagnostics"
    COMPATIBILITY = "compatibility"
    TOOLS = "tools"
    CABLES = "cables"
    SUPPORT = "support"
    SCRIPTS = "scripts"
    TABLES = "tables"
    COURSES = "courses"

class TableName(str, Enum):
    """Enum for available table types"""
    GENERATORS_START_STOP = "generators_start-stop"
    MS112_CABLES_FITTINGS = "MS112_cables_and_fittings"
    MS561_PROGRAMS = "MS561_programs"
    MSG_ALTERNATOR_CROSSLIST = "msg_alternator_crosslist"
    MSG_ALTERNATORS = "msg_alternators"


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
    
    tables_to_query: list[TableName] = Field(
        ...,
        description="List of specific tables to query. Only populated when 'tables' specialist is selected, otherwise empty array."
    )
    
    reason: str = Field(
        ..., 
        description="Explanation of why these specific specialists were chosen for the query"
    )
    
    class Config:
        extra = 'forbid'  # Equivalent to additionalProperties: false
        use_enum_values = True  # Use enum values in serialization

    def model_post_init(self, __context) -> None:
        """Custom validation to ensure tables_to_query logic is correct"""
        # If tables specialist is not selected, tables_to_query should be empty
        if SpecialistType.TABLES not in self.specialists and self.tables_to_query:
            raise ValueError("tables_to_query must be empty when 'tables' specialist is not selected")
        
        # Optional: If tables specialist is selected, at least one table should be specified
        # Uncomment if you want to enforce this rule:
        # if SpecialistType.TABLES in self.specialists and not self.tables_to_query:
        #     raise ValueError("At least one table must be specified when 'tables' specialist is selected")