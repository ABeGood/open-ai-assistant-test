from pydantic import BaseModel, Field, field_validator
from .enums import SpecialistType, TableName


class SpecialistRoutingResponse(BaseModel):
    """Pydantic model matching response_schema-2.json format"""
    
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
        
        # If tables specialist is selected, at least one table should be specified
        if SpecialistType.TABLES in self.specialists and not self.tables_to_query:
            raise ValueError("At least one table must be specified when 'tables' specialist is selected")


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

class InterpreterResponse(BaseModel):
    """Pydantic model for result interpreter response validation"""
    
    interpretation: str = Field(
        ..., 
        description="Clear, natural language explanation of what the results show, with cleaned and processed data"
    )
    
    rerun_needed: bool = Field(
        ...,
        description="True if major issues require code rerun (empty results, wrong columns, major logic errors), False if results can be interpreted"
    )
    
    fixes: str = Field(
        ...,
        description="Specific instructions for fixing code issues when rerun_needed=True, empty string when rerun_needed=False"
    )
    
    class Config:
        extra = 'forbid'  # Equivalent to additionalProperties: false
    
    def model_post_init(self, __context) -> None:
        """Custom validation to ensure fixes logic is correct"""
        # If rerun is not needed, fixes should be empty
        if not self.rerun_needed and self.fixes.strip():
            raise ValueError("fixes must be empty when rerun_needed is False")
        
        # If rerun is needed, fixes should not be empty
        if self.rerun_needed and not self.fixes.strip():
            raise ValueError("fixes must contain instructions when rerun_needed is True")
    
    @field_validator('interpretation')
    @classmethod
    def interpretation_not_empty(cls, v: str) -> str:
        """Ensure interpretation is not empty"""
        if not v.strip():
            raise ValueError("interpretation cannot be empty")
        return v
    
def flatten_schema(schema_dict):
    """Remove $defs and inline enum values directly"""
    if '$defs' not in schema_dict:
        return schema_dict
    
    # Extract the definitions
    defs = schema_dict['$defs']
    
    # Create a copy without $defs
    flattened = {k: v for k, v in schema_dict.items() if k != '$defs'}
    
    # Replace $ref with actual enum values
    def replace_refs(obj):
        if isinstance(obj, dict):
            if '$ref' in obj and obj['$ref'].startswith('#/$defs/'):
                ref_name = obj['$ref'].split('/')[-1]
                if ref_name in defs:
                    return defs[ref_name]
                return obj
            return {k: replace_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_refs(item) for item in obj]
        return obj
    
    return replace_refs(flattened)