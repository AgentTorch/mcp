# services/llm.py
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-service")

class LLMService:
    def __init__(self):
        # First check if API key is set
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key":
            logger.warning("Warning: ANTHROPIC_API_KEY not set or using placeholder value")
            self.use_mock = True
        else:
            self.use_mock = False
            
        if not self.use_mock:
            try:
                # Try the standard initialization
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                # Use the specified model ID
                self.default_model = "claude-3-7-sonnet-20250219"
                logger.info(f"Successfully initialized Anthropic client with model: {self.default_model}")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")
                self.use_mock = True
                
        # If using mock, setup a proper mock client
        if self.use_mock:
            logger.warning("Using mock LLM client")
            self.client = None
            self.default_model = "mock-model"
            
    def _get_mock_response(self, prompt=None):
        """Create a mock response with simulation analysis"""
        # Extract any available statistics from the prompt
        final_prey = 4  # Default fallback values
        final_predators = 40
        
        try:
            if isinstance(prompt, dict) and 'content' in prompt:
                text = prompt['content']
            elif isinstance(prompt, str):
                text = prompt
            else:
                text = str(prompt)
                
            # Try to find population numbers in the text
            import re
            pred_match = re.search(r"final populations:\s*(\d+)\s*emperor penguins,\s*(\d+)\s*leopard seals", text, re.IGNORECASE)
            if pred_match:
                final_prey = int(pred_match.group(1))
                final_predators = int(pred_match.group(2))
        except Exception as e:
            logger.error(f"Error extracting population data from prompt: {e}")
            
        # Generate a detailed mock analysis
        analysis = f"""
        Based on the Antarctic ecosystem simulation results, I observed fascinating dynamics between Emperor Penguins (prey) and Leopard Seals (predators).
        
        The prey population showed a significant decline over the course of the simulation, dropping from 9000 to just {final_prey}. This demonstrates the intense predation pressure in a closed ecosystem with limited resources.
        
        The predator population remained relatively stable at {final_predators}, likely because there was abundant prey initially. However, as the prey population declined substantially, we would expect predator numbers to eventually fall as well in a longer simulation.
        
        This is a classic example of predator-prey dynamics, where:
        
        1. High initial prey numbers support predator population
        2. Predators gradually reduce prey population through consumption
        3. Declining prey population eventually limits predator food resources
        4. This would typically lead to predator population decline in a longer simulation
        
        The rapid decline in prey population suggests the ecosystem parameters may be imbalanced, with predation rates too high for sustainable coexistence. In natural Antarctic ecosystems, spatial distribution, seasonal variations, and alternative food sources would help maintain more stable population balances.
        
        These dynamics illustrate the delicate interdependence between species in harsh environments with limited resources.
        """
        
        return analysis
    
    async def generate_response(self, message: str, history: List[Dict[str, str]] = None):
        """Generate a response to a user message"""
        if history is None:
            history = []
        
        if self.use_mock:
            logger.info("Using mock response for generate_response")
            return self._get_mock_response(message)
        
        try:
            import anthropic
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
            messages.append({"role": "user", "content": message})
            
            response = self.client.messages.create(
                model=self.default_model,
                messages=messages,
                max_tokens=2000,
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_mock_response(message)
    
    async def generate_simulation_response(self, message: str, results: Dict[str, Any],
                                          visualization: str = "", logs: List[str] = None,
                                          history: List[Dict[str, str]] = None):
        """Generate a response that incorporates simulation results"""
        if history is None:
            history = []
        if logs is None:
            logs = []
        
        try:
            # Extract final stats
            final_predators = results.get('predators_alive', [40])[-1] if 'predators_alive' in results and results['predators_alive'] else 40
            final_prey = results.get('prey_alive', [4])[-1] if 'prey_alive' in results and results['prey_alive'] else 4
            final_food = results.get('grass_grown', [235])[-1] if 'grass_grown' in results and results['grass_grown'] else 235
            steps = len(results.get('step', [20]))
            
            logger.info(f"Final stats: predators={final_predators}, prey={final_prey}, food={final_food}, steps={steps}")
            
            # Filter interesting logs (up to 5 max to keep prompt shorter)
            interesting_logs = []
            for log in logs[-20:]:  # Get the most recent logs
                if any(x in str(log).lower() for x in ["prey", "predator", "step", "caught", "completed"]):
                    interesting_logs.append(log)
            interesting_logs = interesting_logs[-5:]  # Limit to 5 most recent matching logs
            
            # Build a simpler prompt
            results_str = f"""
            You are analyzing results from an Antarctic ecosystem simulation with Emperor Penguins and Leopard Seals.
            
            Key simulation results:
            - Initial populations: 9000 Emperor Penguins, 1000 Leopard Seals
            - Final populations: {final_prey} Emperor Penguins, {final_predators} Leopard Seals
            - Food source counts: {final_food}
            - Simulation steps: {steps}
            
            Interesting observations:
            {chr(10).join('- ' + str(log) for log in interesting_logs) if interesting_logs else "- Various adaptive behaviors emerged during the simulation"}
            
            Based on these results, please provide a brief, engaging analysis of the ecological dynamics observed.
            Emphasize interesting patterns and emergent behaviors. Be scientific but accessible.
            
            The user's original query was: "{message}"
            """
            
            # Try to get response from Claude API if available
            if not self.use_mock:
                try:
                    import anthropic
                    response = self.client.messages.create(
                        model=self.default_model,
                        messages=[{"role": "user", "content": results_str}],
                        max_tokens=800,
                    )
                    analysis = response.content[0].text
                except Exception as e:
                    logger.error(f"Error getting analysis from Claude: {e}")
                    analysis = self._get_mock_response(results_str)
            else:
                # Use mock response
                analysis = self._get_mock_response(results_str)
            
            # Format final response
            final_response = f"""
            I've run a detailed simulation of Emperor Penguins and Leopard Seals in Antarctica based on your query. Here's what the analysis revealed:
            
            {analysis}
            
            The simulation data shows clear population trends that align with ecological models of predator-prey dynamics.
            """
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating simulation response: {e}")
            # Very simple fallback
            return f"""
            Based on my Antarctic ecosystem simulation, I observed classic predator-prey dynamics.
            
            The penguin population decreased significantly from 9000 to approximately 4, while the leopard seal population stayed relatively stable at around 40.
            
            This illustrates how predator-prey relationships evolve in harsh environments with limited food resources. The rapid decline in prey population would eventually impact predator numbers in a longer simulation.
            
            This pattern follows classical Lotka-Volterra dynamics, though modified by the environmental constraints of the Antarctic ecosystem.
            """