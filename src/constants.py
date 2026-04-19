"""Enums and constants for the application."""

from enum import Enum


class TransitMode(str, Enum):
    """Transit mode options."""
    HRT = "HRT"  # Heavy Rail Transit
    LRT = "LRT"  # Light Rail Transit
    BRT = "BRT"  # Bus Rapid Transit
    CRT = "CRT"  # Commuter Rail Transit


class SubsystemLabel(str, Enum):
    """Subsystem cost category labels."""
    GUIDEWAY = "Guideway"
    STATIONS = "Stations"
    SYSTEMS = "Systems"
    SOFT_COSTS = "Soft Costs"
    VEHICLES = "Vehicles"
    ROW = "Right-of-Way (ROW)"
    SITEWORK = "Sitework"
    FACILITIES = "Facilities"
