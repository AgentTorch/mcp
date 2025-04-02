# routers/simulation.py
from fastapi import APIRouter, HTTPException, Body
from models.nlp import SimulationRequest, SubstepRequest
from controller import Controller

router = APIRouter(
    prefix="/api/simulation",
    tags=["simulation"],
)

controller = Controller()

@router.post("/run")
async def run_simulation(request: SimulationRequest):
    """Run a simulation based on natural language description"""
    try:
        # Extract simulation parameters
        params = {
            "model_type": request.model_type,
            "description": request.description,
            "steps": request.steps
        }
        
        # Run simulation
        results, visualization, logs = await controller.simulation_service.run_simulation(
            model_type=params["model_type"],
            config_params=params,
            steps=params["steps"]
        )
        
        return {
            "results": results,
            "visualization": visualization,
            "description": request.description,
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/substep")
async def create_substep(request: SubstepRequest):
    """Create a new substep based on natural language description"""
    try:
        result = await controller.process_substep_creation(
            description=request.description,
            model_type=request.model_type
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))