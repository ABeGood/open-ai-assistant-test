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