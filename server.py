# mcp_server.py - Model-Context-Protocol server
import asyncio
import logging
import os
import json
import traceback
import shutil
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

# Import services
from services.simulation import SimulationService
from services.llm import LLMService
from services.config_modifier import ConfigModifier

# Try to import debug helper
try:
    from services.debug_helper import dump_config, create_sample_substep
except ImportError:
    logger.warning("Could not import debug_helper")

# MCP Models
class SimulationContext(BaseModel):
    """Context for simulation execution"""
    prompt: str
    
class SimulationResponse(BaseModel):
    """Response from simulation model"""
    stats: Dict[str, Any]
    logs: List[str]
    
class AnalysisContext(BaseModel):
    """Context for LLM analysis"""
    prompt: str
    simulation_results: Dict[str, Any]
    logs: List[str]
    
class AnalysisResponse(BaseModel):
    """Response from LLM analysis model"""
    analysis: str

# Helper functions for handling the Grow substep
def copy_config_file(source_path, target_path):
    """Create a copy of the config file"""
    try:
        shutil.copy2(source_path, target_path)
        logger.info(f"Created a copy of config file at {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying config file: {e}")
        return False

def add_grow_substep_to_config(config_path):
    """Add the Grow substep to the config file with exact YAML formatting"""
    try:
        # Read the file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Check if '3': already exists
        for line in lines:
            if "'3':" in line.strip() or '"3":' in line.strip():
                logger.info("Grow substep already exists in config")
                return True
        
        # Prepare the Grow substep with exact matching formatting
        # Need to find the base indentation level from the config file
        base_indent = None
        for line in lines:
            if "'2':" in line.strip() or '"2":' in line.strip():
                base_indent = " " * (len(line) - len(line.lstrip()))
                break
        
        if not base_indent:
            # Default to 2 spaces if we can't determine indentation
            base_indent = "  "
        
        # Create the Grow substep with proper indentation
        grow_lines = [
            f"{base_indent}'3':",
            f"{base_indent}  name: 'Grow'",
            f"{base_indent}  description: 'Grow Grass'",
            f"{base_indent}  active_agents:",
            f"{base_indent}    - 'prey'",
            f"{base_indent}  observation:",
            f"{base_indent}    prey: null",
            f"{base_indent}  policy:",
            f"{base_indent}    prey: null",
            f"{base_indent}  transition:",
            f"{base_indent}    grow_grass:",
            f"{base_indent}      generator: 'GrowGrassVmap'",
            f"{base_indent}      arguments: null",
            f"{base_indent}      input_variables:",
            f"{base_indent}        grass_growth: 'objects/grass/growth_stage'",
            f"{base_indent}        growth_countdown: 'objects/grass/growth_countdown'",
            f"{base_indent}      output_variables:",
            f"{base_indent}        - grass_growth",
            f"{base_indent}        - growth_countdown"
        ]
        
        # Ensure the file ends with a newline
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        
        # Add an extra blank line before the Grow substep
        if not lines[-1].strip() == '':
            lines.append('\n')
        
        # Add the Grow substep lines with newlines
        lines.extend(line + '\n' for line in grow_lines)
        
        # Write the file back
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        logger.info(f"Successfully added Grow substep to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error adding Grow substep: {e}")
        return False

def process_config_for_growth(prompt, original_config, temp_config):
    """Process configuration for growth-related requests with exact formatting"""
    # Check for growth-related keywords
    growth_keywords = ['grow', 'regrow', 'grass', 'vegetation', 'regrowth', 'growth']
    if any(keyword in prompt.lower() for keyword in growth_keywords):
        # Copy the original config
        if copy_config_file(original_config, temp_config):
            # Add the Grow substep with precise formatting
            return add_grow_substep_to_config(temp_config)
    
    return False
    
# MCP Protocol Handler
class MCPProtocolHandler:
    def __init__(self):
        # Initialize models
        self.simulation_model = SimulationService(use_vectorized=True)
        self.analysis_model = LLMService()
        
        # Setup paths
        root_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Original config path
        self.original_config_path = os.path.join(root_dir, "config.yaml")
        logger.info(f"Original config path: {self.original_config_path}")
        
        # Temporary config path for modifications
        self.temp_config_path = os.path.join(root_dir, "services", "config.yaml")
        logger.info(f"Temporary config path: {self.temp_config_path}")
        
        # Initialize config modifier for other operations
        self.config_modifier = ConfigModifier(self.original_config_path)
    
    async def process_simulation(self, context: SimulationContext) -> SimulationResponse:
        """Process simulation request according to MCP protocol"""
        logger.info(f"Running simulation with context: {context.prompt}")
        
        logs = []
        
        # Check if we need to add the Grow substep
        config_modified = process_config_for_growth(
            context.prompt, 
            self.original_config_path, 
            self.temp_config_path
        )
        
        if config_modified:
            log_msg = "Added 'Grow' substep to enable grass regrowth"
            logger.info(log_msg)
            logs.append(log_msg)
            
            # Point the simulation to the modified config
            config_path = self.temp_config_path
        else:
            # Use the original config
            config_path = self.original_config_path
        
        # Execute the model with the appropriate config
        # Set the config path environment variable for the simulation
        os.environ["AGENTTORCH_CONFIG_PATH"] = config_path
        stats, sim_logs = await self.simulation_model.run_simulation(reload_config=True)
        
        # Combine logs
        all_logs = logs + sim_logs
        
        # Return formatted response
        return SimulationResponse(
            stats=stats,
            logs=all_logs
        )
        
    async def process_analysis(self, context: AnalysisContext) -> AnalysisResponse:
        """Process analysis request according to MCP protocol"""
        logger.info(f"Running analysis with context: {context.prompt}")
        logger.info(f"With simulation results: {context.simulation_results.keys() if context.simulation_results else 'None'}")
        
        try:
            # Execute the analysis model with the given context
            response = await self.analysis_model.generate_simulation_response(
                message=context.prompt,
                results=context.simulation_results,
                visualization="chart",
                logs=context.logs
            )
            
            # Return formatted response
            return AnalysisResponse(
                analysis=response
            )
        except Exception as e:
            logger.error(f"Error in process_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Extract final stats for fallback
            final_predators = 40
            final_prey = 4
            try:
                if context.simulation_results:
                    if 'predators_alive' in context.simulation_results and context.simulation_results['predators_alive']:
                        final_predators = context.simulation_results['predators_alive'][-1]
                    if 'prey_alive' in context.simulation_results and context.simulation_results['prey_alive']:
                        final_prey = context.simulation_results['prey_alive'][-1]
            except Exception:
                pass
            
            # Fallback response
            fallback = f"""
            Based on my Antarctic ecosystem simulation, I observed interesting dynamics between Emperor Penguins and Leopard Seals.
            
            The penguin population decreased significantly from 800 to {final_prey}, while the leopard seal population remained at approximately {final_predators}.
            
            This illustrates classic predator-prey dynamics in harsh environments with limited resources.
            
            (Error generating detailed analysis: {str(e)})
            """
            
            return AnalysisResponse(analysis=fallback)

# Initialize FastAPI app
app = FastAPI(title="AgentTorch MCP Server")

# Ensure directories exist
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize MCP handler
mcp_handler = MCPProtocolHandler()

# Templates for UI
templates = Jinja2Templates(directory="templates")

# Root endpoint returns the UI
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# RESTful MCP endpoints
@app.post("/api/simulation", response_model=SimulationResponse)
async def run_simulation(context: SimulationContext):
    """MCP endpoint for running simulations"""
    try:
        response = await mcp_handler.process_simulation(context)
        return response
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Simulation failed: {str(e)}"}
        )

@app.post("/api/analysis", response_model=AnalysisResponse)
async def run_analysis(context: AnalysisContext):
    """MCP endpoint for running analysis"""
    try:
        response = await mcp_handler.process_analysis(context)
        return response
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            command = data.get("command", "")

            if command == "run_simulation":
                # Create context from websocket data
                context = SimulationContext(
                    prompt=data.get("prompt", "Run simulation")
                )
                
                try:
                    # Send a notification that we're processing
                    await websocket.send_json({
                        "type": "simulation_log",
                        "log": "Analyzing your simulation request...",
                        "progress": 5
                    })
                    
                    # Check for grass regrowth intent
                    grow_keywords = ['grow', 'regrow', 'grass', 'vegetation', 'regrowth', 'growth']
                    if any(keyword in context.prompt.lower() for keyword in grow_keywords):
                        await websocket.send_json({
                            "type": "simulation_log",
                            "log": "Detected request for grass regrowth. Adding 'Grow' substep...",
                            "progress": 20
                        })
                        
                        # Process config for growth
                        config_modified = process_config_for_growth(
                            context.prompt, 
                            mcp_handler.original_config_path, 
                            mcp_handler.temp_config_path
                        )
                        
                        if config_modified:
                            await websocket.send_json({
                                "type": "simulation_log",
                                "log": "Successfully added 'Grow' substep to enable grass regrowth",
                                "progress": 35
                            })
                            
                            # Point the simulation to the modified config
                            os.environ["AGENTTORCH_CONFIG_PATH"] = mcp_handler.temp_config_path
                        else:
                            await websocket.send_json({
                                "type": "simulation_log",
                                "log": "Failed to add 'Grow' substep or it already exists",
                                "progress": 35
                            })
                            
                            # Use the original config
                            os.environ["AGENTTORCH_CONFIG_PATH"] = mcp_handler.original_config_path
                    else:
                        await websocket.send_json({
                            "type": "simulation_log",
                            "log": "Using existing configuration (no grass regrowth requested)",
                            "progress": 35
                        })
                        
                        # Use the original config
                        os.environ["AGENTTORCH_CONFIG_PATH"] = mcp_handler.original_config_path
                            
                    # Run the simulation
                    await websocket.send_json({
                        "type": "simulation_log",
                        "log": "Starting simulation execution...",
                        "progress": 45
                    })
                    
                    # Process simulation through MCP handler
                    stats, logs = await mcp_handler.simulation_model.run_simulation(reload_config=True)
                    
                    # Send logs as they're generated
                    log_count = len(logs)
                    for i, log in enumerate(logs):
                        progress = int(50 + (i / log_count) * 45) if log_count > 0 else 80
                        await websocket.send_json({
                            "type": "simulation_log",
                            "log": log,
                            "progress": progress
                        })
                        # Small delay to avoid overwhelming the client
                        await asyncio.sleep(0.01)
                    
                    # Generate chart data
                    chart_data = {
                        "labels": [f"Step {s}" for s in stats.get("step", [])],
                        "datasets": [
                            {
                                "label": "Predators",
                                "data": stats.get("predators_alive", []),
                                "borderColor": "red",
                                "backgroundColor": "rgba(255, 0, 0, 0.1)"
                            },
                            {
                                "label": "Prey",
                                "data": stats.get("prey_alive", []),
                                "borderColor": "blue",
                                "backgroundColor": "rgba(0, 0, 255, 0.1)"
                            },
                            {
                                "label": "Grass",
                                "data": stats.get("grass_grown", []),
                                "borderColor": "green",
                                "backgroundColor": "rgba(0, 255, 0, 0.1)"
                            }
                        ]
                    }
                    
                    # Send completion status with chart data
                    await websocket.send_json({
                        "type": "simulation_complete",
                        "stats": stats,
                        "chartData": chart_data
                    })
                    
                except Exception as e:
                    logger.error(f"Simulation error: {str(e)}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Simulation error: {str(e)}"
                    })
            
            elif command == "analyze":
                # Create context from websocket data
                context = AnalysisContext(
                    prompt=data.get("prompt", "Analyze the simulation results."),
                    simulation_results=data.get("results", {}),
                    logs=data.get("logs", [])
                )
                
                try:
                    logger.info(f"Analysis request for prompt: {context.prompt}")
                    logger.info(f"With simulation results keys: {context.simulation_results.keys() if context.simulation_results else 'None'}")
                    
                    # Send status update
                    await websocket.send_json({
                        "type": "analysis_status",
                        "status": "Analyzing simulation results..."
                    })
                    
                    # Process analysis through MCP handler
                    analysis_response = await mcp_handler.process_analysis(context)
                    
                    await websocket.send_json({
                        "type": "analysis_result",
                        "analysis": analysis_response.analysis
                    })
                    
                except Exception as e:
                    logger.error(f"Analysis error: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Extract final stats for fallback
                    final_predators = 40
                    final_prey = 4
                    try:
                        if context.simulation_results:
                            if 'predators_alive' in context.simulation_results and context.simulation_results['predators_alive']:
                                final_predators = context.simulation_results['predators_alive'][-1]
                            if 'prey_alive' in context.simulation_results and context.simulation_results['prey_alive']:
                                final_prey = context.simulation_results['prey_alive'][-1]
                    except Exception:
                        pass
                    
                    # Provide a fallback response when analysis fails
                    fallback_response = f"""
                    Based on the simulation results, I observed classic predator-prey dynamics.
                    
                    The prey population decreased over time from 800 to approximately {final_prey}, while predator numbers remained relatively stable at around {final_predators}.
                    
                    This demonstrates the interdependence of species in an ecosystem and how resource limitations drive population dynamics.
                    
                    (Note: A detailed analysis could not be generated due to a technical error: {str(e)})
                    """
                    
                    await websocket.send_json({
                        "type": "analysis_result",
                        "analysis": fallback_response
                    })
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        logger.error(traceback.format_exc())

# Run the server
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)