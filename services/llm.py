# services/llm.py
import os
from typing import List, Dict, Any
import anthropic

class LLMService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
        self.default_model = "claude-3-7-sonnet-20240229"
    
    async def generate_response(self, message: str, history: List[Dict[str, str]] = None):
        """Generate a response to a user message"""
        if history is None:
            history = []
        
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.append({"role": "user", "content": message})
        
        response = self.client.messages.create(
            model=self.default_model,
            messages=messages,
            max_tokens=2000,
        )
        
        return response.content[0].text
    
    async def generate_simulation_response(self, message: str, results: Dict[str, Any], 
                                          visualization: str, logs: List[str], 
                                          history: List[Dict[str, str]] = None):
        """Generate a response that incorporates simulation results"""
        if history is None:
            history = []
        
        # Create a prompt that includes the simulation results
        results_str = f"""
        Simulation Results:
        - Steps Run: {len(results['step'])}
        - Final Population Counts:
          - Predators (Leopard Seals): {results['predators_alive'][-1] if 'predators_alive' in results else 'N/A'}
          - Prey (Emperor Penguins): {results['prey_alive'][-1] if 'prey_alive' in results else 'N/A'}
          - Food Sources: {results['grass_grown'][-1] if 'grass_grown' in results else 'N/A'}
        
        Key Events from Logs:
        {logs[-min(5, len(logs)):]}
        
        Please analyze these simulation results and explain what they show about the user's query: "{message}"
        Treat predators as "leopard seals" and prey as "emperor penguins" in your response.
        Include specific insights about population dynamics and explain the key patterns observed.
        Remember, this is modeling Antarctic penguin populations.
        Keep your response concise and focused on the most interesting insights.
        """
        
        # Get analysis from Claude
        analysis_response = self.client.messages.create(
            model=self.default_model,
            messages=[{"role": "user", "content": results_str}],
            max_tokens=1000,
        )
        
        analysis = analysis_response.content[0].text
        
        # Format final response with image reference
        final_response = f"""
        I've run a simulation of Emperor Penguins and Leopard Seals in Antarctica based on your query. Here's what I found:
        
        {analysis}
        
        [Visualization of simulation results]
        """
        
        return final_response
    
    async def generate_substep_code(self, description: str, model_type: str):
        """Generate code for a new substep from natural language description"""
        prompt = f"""
        Generate PyTorch code for a new substep in the {model_type} model based on this description:
        
        {description}
        
        For the Antarctic ecosystem simulation:
        - "predator" agents represent Leopard Seals
        - "prey" agents represent Emperor Penguins
        - "grass" objects represent Krill/Fish food sources
        
        The code should follow the AgentTorch framework format with:
        1. A class extending SubstepAction, SubstepObservation, or SubstepTransition
        2. Registration with the @Registry.register_substep decorator
        3. Implementation of the forward method
        4. Proper handling of state and input/output variables
        
        Return ONLY the Python code without any explanations or markdown formatting.
        """
        
        response = self.client.messages.create(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        
        return response.content[0].text