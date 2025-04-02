# models/simulation.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AgentStats(BaseModel):
    step: List[int]
    predators_alive: Optional[List[int]] = None
    prey_alive: Optional[List[int]] = None
    grass_grown: Optional[List[int]] = None

class SimulationResult(BaseModel):
    stats: AgentStats
    visualization: str  # Base64 encoded image/gif
    description: str
    config: Dict[str, Any]