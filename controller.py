# controller.py
from services.llm import LLMService
from services.nlp_parser import NLPParser
from services.simulation import SimulationService

class Controller:
    def __init__(self):
        self.llm_service = LLMService()
        self.nlp_parser = NLPParser()
        self.simulation_service = SimulationService()
    
    async def process_chat_request(self, message, history=None):
        """Process a chat request and determine if simulation is needed"""
        if history is None:
            history = []
        
        # Check if message requires simulation
        sim_request = self.nlp_parser.detect_simulation_request(message)
        
        if sim_request:
            # Extract simulation parameters
            params = self.nlp_parser.extract_simulation_params(message)
            
            # Add logging for substeps
            logs = []
            def log_callback(step, substep, agent_type, action):
                log_entry = f"Step {step}, Substep {substep}: {agent_type} {action}"
                logs.append(log_entry)
                return log_entry
            
            # Run simulation
            results, visualization, execution_logs = await self.simulation_service.run_simulation(
                model_type=params.get("model_type", "predator_prey"),
                config_params=params,
                steps=params.get("steps", 20),
                log_callback=log_callback
            )
            
            # Generate response incorporating simulation results
            response = await self.llm_service.generate_simulation_response(
                message, results, visualization, execution_logs, history
            )
        else:
            # Regular chat response
            response = await self.llm_service.generate_response(message, history)
        
        return response
    
    async def process_substep_creation(self, description, model_type="predator_prey"):
        """Process a request to create a new substep"""
        # Generate substep code from description
        substep_code = await self.llm_service.generate_substep_code(description, model_type)
        
        # Register the new substep
        success, message = await self.simulation_service.register_new_substep(substep_code, model_type)
        
        return {
            "success": success,
            "message": message,
            "code": substep_code,
            "description": description
        }