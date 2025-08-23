from pydantic import BaseModel, Field
from typing import List, Optional, Any
from enum import Enum


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool
    status: ResponseStatus
    error: Optional[str] = None
    user_query: str
    timestamp: Optional[float] = None
    
    class Config:
        use_enum_values = True


class OrchestratorResponse(BaseResponse):
    """Response from orchestrator routing decision"""
    specialists: List[str] = Field(default_factory=list)
    tables_to_query: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    confidence: Optional[float] = None
    raw_response: Optional[str] = None


class SpecialistResponse(BaseResponse):
    """Response from individual specialist assistant"""
    specialist: str
    response: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None


class CombinatorResponse(BaseResponse):
    """Response from combinator that merges specialist responses"""
    specialists: List[str] = Field(default_factory=list)
    response: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    raw_response: Optional[str] = None
    combined_from: List[SpecialistResponse] = Field(default_factory=list)


class MultiSpecialistResponse(BaseResponse):
    """Response from calling multiple specialists"""
    successful_responses: List[SpecialistResponse] = Field(default_factory=list)
    failed_responses: List[SpecialistResponse] = Field(default_factory=list)
    total_specialists: int = 0
    success_rate: Optional[float] = None


# Helper functions for creating responses
def create_error_response(
    response_type: type,
    error_message: str,
    user_query: str,
    **kwargs
) -> BaseResponse:
    """Create a standardized error response"""
    base_data = {
        "success": False,
        "status": ResponseStatus.ERROR,
        "error": error_message,
        "user_query": user_query
    }
    base_data.update(kwargs)
    return response_type(**base_data)


def create_success_response(
    response_type: type,
    user_query: str,
    **kwargs
) -> BaseResponse:
    """Create a standardized success response"""
    base_data = {
        "success": True,
        "status": ResponseStatus.SUCCESS,
        "error": None,
        "user_query": user_query
    }
    base_data.update(kwargs)
    return response_type(**base_data)


def create_timeout_response(
    response_type: type,
    user_query: str,
    timeout_message: str = "Operation timed out",
    **kwargs
) -> BaseResponse:
    """Create a standardized timeout response"""
    base_data = {
        "success": False,
        "status": ResponseStatus.TIMEOUT,
        "error": timeout_message,
        "user_query": user_query
    }
    base_data.update(kwargs)
    return response_type(**base_data)