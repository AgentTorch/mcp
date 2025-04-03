# controller.py
from services.llm import LLMService
from services.nlp_parser import NLPParser
from services.simulation import SimulationService
import logging

logger = logging.getLogger("agenttorch")

class Controller:
    def __init__(self):
        self.llm_service = LLMService()
        self.nlp_parser = NLPParser()
        self.simulation_service = SimulationService()
    
    async def process_chat_request(self, message, history=None):
        """Process a chat request and determine if simulation is needed"""
        if history is None:
            history = []
        
        simulation_data = None
        
        try:
            # Check if message requires simulation
            sim_request = self.nlp_parser.detect_simulation_request(message)
            
            if sim_request:
                logger.info(f"Simulation request detected: {message}")
                # Extract simulation parameters
                try:
                    params = self.nlp_parser.extract_simulation_params(message)
                    logger.info(f"Extracted parameters: {params}")
                except Exception as e:
                    logger.error(f"Error extracting simulation parameters: {e}")
                    # Default parameters if extraction fails
                    params = {
                        "model_type": "predator_prey",
                        "steps": 30,
                        "config_params": {
                            "num_predators": 5000,
                            "num_prey": 25000,
                            "num_grass": 10000
                        }
                    }
                
                # Add logging for substeps
                logs = []
                def log_callback(step, substep, agent_type, action):
                    log_entry = f"Step {step}, Substep {substep}: {agent_type} {action}"
                    logs.append(log_entry)
                    logger.info(log_entry)
                    return log_entry
                
                # Run simulation
                try:
                    results, visualization, execution_logs = await self.simulation_service.run_simulation(
                        model_type=params.get("model_type", "predator_prey"),
                        config_params=params,
                        steps=params.get("steps", 30),
                        log_callback=log_callback
                    )
                    
                    # Save simulation data to return with response
                    simulation_data = {
                        "results": results,
                        "visualization": visualization,
                        "logs": execution_logs
                    }
                    
                    logger.info(f"Simulation completed: {len(execution_logs)} log entries, visualization: {len(visualization)} bytes")
                except Exception as e:
                    logger.error(f"Error running simulation: {e}")
                    # Fallback if simulation fails
                    results = {
                        "step": list(range(30)),
                        "predators_alive": [5000] * 30,
                        "prey_alive": [25000] * 30,
                        "grass_grown": [10000] * 30
                    }
                    visualization = ""
                    execution_logs = [f"Simulation error: {str(e)}"]
                    
                    # Still provide partial simulation data
                    simulation_data = {
                        "results": results,
                        "visualization": visualization,
                        "logs": execution_logs
                    }
                
                # Generate response incorporating simulation results
                response = await self.llm_service.generate_simulation_response(
                    message, results, visualization, execution_logs, history
                )
            else:
                # Regular chat response
                logger.info(f"Regular chat request: {message}")
                response = await self.llm_service.generate_response(message, history)
        except Exception as e:
            logger.error(f"Error in process_chat_request: {e}")
            # Ultimate fallback
            response = f"""
            I apologize, but I encountered an error while processing your request. 
            
            I'd be happy to discuss Antarctic penguin ecosystems based on my knowledge. Emperor penguins have fascinating adaptations for surviving in harsh conditions, including huddling behaviors to conserve heat and specialized hunting techniques.
            
            Would you like to try your question again or ask something different about Antarctic ecosystems?
            """
        
        return response, simulation_data
    
    async def process_substep_creation(self, description, model_type="predator_prey"):
        """Process a request to create a new substep"""
        try:
            logger.info(f"Creating new substep: {description}")
            # Generate substep code from description
            substep_code = await self.llm_service.generate_substep_code(description, model_type)
            
            # Register the new substep
            success, message = await self.simulation_service.register_new_substep(substep_code, model_type)
            
            logger.info(f"Substep creation result: {success}, {message}")
            
            return {
                "success": success,
                "message": message,
                "code": substep_code,
                "description": description
            }
        except Exception as e:
            logger.error(f"Error creating substep: {e}")
            return {
                "success": False,
                "message": f"Error creating substep: {str(e)}",
                "code": "",
                "description": description
            }