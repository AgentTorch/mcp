# models/nlp.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "claude-3-7-sonnet-20240229"
    temperature: Optional[float] = 0.7

class SimulationParams(BaseModel):
    model_type: str = "predator_prey"
    steps: int = 20
    config_params: Dict[str, Any] = {}
    description: str = ""

class SimulationRequest(BaseModel):
    description: str
    model_type: Optional[str] = "predator_prey"
    steps: Optional[int] = 20

class SubstepRequest(BaseModel):
    description: str
    model_type: Optional[str] = "predator_prey"