# services/nlp_parser.py
import re
from typing import Dict, Any
import anthropic
import os
import json

class NLPParser:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
    
    def detect_simulation_request(self, message: str) -> bool:
        """Detect if a message is requesting a simulation"""
        simulation_keywords = [
            r"simulat(e|ion)",
            r"model",
            r"run .*(scenario|experiment)",
            r"what (would|if) happen",
            r"show me",
            r"visualize",
            r"compare .* (behaviors|strategies)",
            r"penguins?",
            r"seals?",
            r"antarctic(a)?",
        ]
        
        for keyword in simulation_keywords:
            if re.search(keyword, message, re.IGNORECASE):
                return True
        
        return False
    
    def extract_simulation_params(self, message: str) -> Dict[str, Any]:
        """Extract simulation parameters from a natural language message"""
        prompt = f"""
        Parse the following message into parameters for an Antarctic ecosystem simulation.
        
        Message: {message}
        
        Extract the following information in JSON format:
        1. model_type: The type of model to use (predator_prey)
        2. steps: Number of simulation steps to run
        3. config_params: A dictionary with configuration parameters including:
           - agent counts (num_predators, num_prey - these represent leopard seals and emperor penguins)
           - environmental factors (temperature, wind_speed, etc.)
           - behavioral parameters (energy, movement_patterns, etc.)
        
        If a parameter is not specified, provide sensible defaults for an Antarctic ecosystem.
        Use the predator_prey model as the base, but adapt it to represent an Antarctic ecosystem with
        Leopard Seals (predators) and Emperor Penguins (prey).
        """
        
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20240229",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        
        # Extract JSON from response
        response_text = response.content[0].text
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_str = response_text
        
        try:
            params = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback to default parameters for Antarctic simulation
            params = {
                "model_type": "predator_prey",
                "steps": 30,
                "config_params": {
                    "num_predators": 1000,  # Leopard seals
                    "num_prey": 9000,       # Emperor penguins
                    "num_grass": 5000,      # Krill/Fish patches
                    "environment": {
                        "temperature": -20, # Celsius
                        "wind_speed": 30,   # km/h
                    }
                }
            }
        
        return params