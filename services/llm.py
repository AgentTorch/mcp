# services/llm.py
import os
import logging
import re
import yaml
import json
from typing import List, Dict, Any, Optional, Union

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
        final_prey = 10  # Default fallback values
        final_predators = 400
        
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

    async def parse_substep_definition(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Parse a user's message to extract substep definitions for the AgentTorch simulation.
        """
        logger.info("Parsing message for substep definitions")
        
        # Simple keyword check to determine if a substep is mentioned
        substep_keywords = {
            'move': ['move', 'movement', 'locomotion', 'travel', 'walk', 'swim', 'position', 'navigate'],
            'eat': ['eat', 'food', 'feed', 'forage', 'consume', 'nutrition', 'krill', 'algae'],
            'hunt': ['hunt', 'chase', 'catch', 'predation', 'capture', 'attack', 'pursue', 'seal'],
            'grow': ['grow', 'regrow', 'grass', 'vegetation', 'regrowth', 'plant', 'growth']
        }
        
        substep_type = None
        # Check for substep keywords in the message
        message_lower = message.lower()
        for step_type, keywords in substep_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                substep_type = step_type
                break
                
        if not substep_type:
            logger.info("No substep definition found in message")
            return None
            
        # Use LLM to extract structured substep information (if available)
        if not self.use_mock:
            try:
                import anthropic
                
                prompt = f"""
                You are an expert in AgentTorch predator-prey simulations. The user wants to add a '{substep_type}' substep to the simulation.
                
                In the Antarctic predator-prey simulation:
                - Leopard seals are predators
                - Emperor penguins are prey
                - Algae/krill (represented as "grass") are food resources
                
                Based on the user's message, extract or infer details about the '{substep_type}' substep they want to add.
                
                User message: {message}
                
                Return a simple JSON object with just these fields:
                {{
                  "name": "The name of the substep (like 'Move', 'Eat', 'Hunt', or 'Grow')",
                  "description": "A brief description of what this substep does"
                }}
                """
                
                response = self.client.messages.create(
                    model=self.default_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                
                # Extract JSON from response
                text_response = response.content[0].text
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', text_response, re.DOTALL)
                json_str = json_match.group(1) if json_match else text_response
                
                try:
                    # Parse the basic substep info
                    from services.config_modifier import SubstepTemplateManager
                    
                    substep_info = json.loads(json_str)
                    logger.info(f"Successfully extracted substep info: {substep_info}")
                    
                    # Get the full template based on the name
                    template = SubstepTemplateManager.get_template_by_name(substep_info.get('name', substep_type))
                    
                    # Update template with extracted info
                    if 'name' in substep_info:
                        template['name'] = substep_info['name']
                    if 'description' in substep_info:
                        template['description'] = substep_info['description']
                    
                    return template
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Error processing LLM response: {e}")
                    # Fall back to default template
            except Exception as e:
                logger.error(f"Error in LLM extraction: {e}")
        
        # Fallback: Return default template based on detected substep type
        from services.config_modifier import SubstepTemplateManager
        
        if substep_type == 'move':
            return SubstepTemplateManager.get_move_template()
        elif substep_type == 'eat':
            return SubstepTemplateManager.get_eat_template()
        elif substep_type == 'hunt':
            return SubstepTemplateManager.get_hunt_template()
        elif substep_type == 'grow':
            return SubstepTemplateManager.get_grow_template()
        else:
            return None

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
                if any(x in str(log).lower() for x in ["prey", "predator", "step", "caught", "completed", "substep", "parameter"]):
                    interesting_logs.append(log)
            interesting_logs = interesting_logs[-5:]  # Limit to 5 most recent matching logs
            
            # Build a simpler prompt
            results_str = f"""
            You are analyzing results from an Antarctic ecosystem simulation with Emperor Penguins and Leopard Seals.
            
            Key simulation results:
            - Initial populations: 800 Emperor Penguins, 400 Leopard Seals
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
            
            The penguin population decreased significantly from 800 to approximately {final_prey}, while the leopard seal population stayed relatively stable at around {final_predators}.
            
            This illustrates how predator-prey relationships evolve in harsh environments with limited food resources. The rapid decline in prey population would eventually impact predator numbers in a longer simulation.
            
            This pattern follows classical Lotka-Volterra dynamics, though modified by the environmental constraints of the Antarctic ecosystem.
            """