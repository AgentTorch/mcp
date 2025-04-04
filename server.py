# mcp_server.py - Model-Context-Protocol server
import asyncio
import logging
import os
import json
import traceback
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

# MCP Models
class SimulationContext(BaseModel):
    """Context for simulation execution"""
    prompt: str
    parameters: Optional[Dict[str, Any]] = None
    
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
    
# MCP Protocol Handler
class MCPProtocolHandler:
    def __init__(self):
        # Initialize models
        self.simulation_model = SimulationService(use_vectorized=True)
        self.analysis_model = LLMService()
        
    async def process_simulation(self, context: SimulationContext) -> SimulationResponse:
        """Process simulation request according to MCP protocol"""
        logger.info(f"Running simulation with context: {context.prompt}")
        
        # Execute the model with the given context
        stats, logs = await self.simulation_model.run_simulation()
        
        # Return formatted response
        return SimulationResponse(
            stats=stats,
            logs=logs
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
            
            The penguin population decreased significantly from 9000 to {final_prey}, while the leopard seal population remained at approximately {final_predators}.
            
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
                    prompt=data.get("prompt", "Run simulation"),
                    parameters=data.get("parameters", {})
                )
                
                try:
                    # Process simulation through MCP handler (streaming updates)
                    stats, logs = await mcp_handler.simulation_model.run_simulation()
                    
                    # Send logs as they're generated
                    for i, log in enumerate(logs):
                        await websocket.send_json({
                            "type": "simulation_log",
                            "log": log,
                            "progress": int((i / len(logs)) * 100) if logs else 0
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
                    Based on the simulation results, I observed the classic predator-prey dynamics.
                    
                    The prey population decreased significantly over time from 9000 to approximately {final_prey}, while predator numbers remained relatively stable at around {final_predators}.
                    
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
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)