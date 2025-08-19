"""
Agents package for the customer support system.
Contains orchestrator and specialist agent implementations.
"""

from .orchestrator_for_tg import create_orchestrator
from .agents_config import *

__all__ = [
    'create_orchestrator'
]