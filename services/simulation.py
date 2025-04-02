# services/simulation.py
import os
import tempfile
import base64
import json
import time
from typing import Dict, Any, Tuple, List, Callable, Optional
import torch
import importlib
import inspect
import sys
import matplotlib.pyplot as plt
import numpy as np

# Try to import AgentTorch components
try:
    from agent_torch.core import Registry, VectorizedRunner
    from agent_torch.core.helpers import read_config, read_from_file
    from agent_torch.core.helpers.environment import grid_network
    from agent_torch.models.predator_prey.helpers.map import map_network
    from agent_torch.models.predator_prey.helpers.random import random_float, random_int
    from agent_torch.models.predator_prey.substeps import *
    from agent_torch.models.predator_prey.vmap_substeps import * 
    from agent_torch.models.predator_prey.plot import Plot
    from agent_torch.core.substep import SubstepObservation, SubstepAction, SubstepTransition
except ImportError:
    print("Warning: Could not import AgentTorch. Make sure it's installed and in your PYTHONPATH.")
    # Create mock classes for testing
    class Registry:
        def __init__(self):
            self.helpers = {"initialization": {}, "network": {}}
        def register(self, func, name, key):
            self.helpers[key][name] = func

    class VectorizedRunner:
        def __init__(self, config, registry):
            self.config = config
            self.registry = registry
            self.state = {"current_step": 0, "agents": {"predator": {"energy": torch.ones(10)}, "prey": {"energy": torch.ones(10)}}}
        def init(self):
            pass
        def step(self, n):
            self.state["current_step"] += n

    class SubstepObservation: pass
    class SubstepAction: pass
    class SubstepTransition: pass

    def read_config(path):
        return {"simulation_metadata": {"num_substeps_per_step": 4, "max_x": 100, "max_y": 100}}

    def read_from_file(shape, params):
        return torch.ones(shape)

    def grid_network(params):
        return None, torch.eye(10)

    def map_network(params):
        return None, torch.eye(10)

    def random_float(shape, params):
        return torch.rand(shape)

    def random_int(shape, params):
        return torch.randint(0, 10, shape)

    class Plot:
        def __init__(self, max_x, max_y):
            self.max_x = max_x
            self.max_y = max_y
        def capture(self, step, state):
            pass
        def compile(self, episode):
            pass

class SimulationService:
    def __init__(self):
        # Path to predator-prey config, pointing to the new location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(base_dir, "../../config/predator_prey/config.yaml")
        
        self.config_paths = {
            "predator_prey": os.environ.get("PREDATOR_PREY_CONFIG", default_config_path)
        }
        
        # Make sure the config file exists
        if not os.path.exists(self.config_paths["predator_prey"]):
            print(f"Warning: Config file not found at {self.config_paths['predator_prey']}")
            print(f"Current directory: {os.getcwd()}")
            # Try to find the config file
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file == "config.yaml" and "predator_prey" in root:
                        self.config_paths["predator_prey"] = os.path.join(root, file)
                        print(f"Found config at: {self.config_paths['predator_prey']}")
                        break
    
    def setup_registry(self):
        """Set up the registry with all necessary functions."""
        registry = Registry()
        registry.register(self.custom_read_from_file, "read_from_file", "initialization")
        registry.register(grid_network, "grid", key="network")
        registry.register(map_network, "map", key="network")
        registry.register(random_float, "random_float", "initialization")
        registry.register(random_int, "random_int", "initialization")
        return registry
    
    def custom_read_from_file(self, shape, params):
        """Custom file reader that handles relative paths correctly."""
        file_path = params["file_path"]
        
        # Make path absolute if needed
        if not os.path.isabs(file_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "../..", file_path)
        
        if file_path.endswith("csv"):
            import pandas as pd
            data = pd.read_csv(file_path)
        
        data_values = data.values
        assert data_values.shape == tuple(shape), f"Shape mismatch: {data_values.shape} vs expected {shape}"

        data_tensor = torch.tensor(data_values, dtype=torch.float32)
        return data_tensor
    
    def create_custom_visualization(self, stats, temp_dir):
        """Create a custom visualization for the simulation results"""
        plt.figure(figsize=(12, 6))
        
        # Population subplot
        plt.subplot(1, 2, 1)
        plt.plot(stats["step"], stats["predators_alive"], 'r-', label='Leopard Seals')
        plt.plot(stats["step"], stats["prey_alive"], 'b-', label='Emperor Penguins')
        plt.title('Antarctic Population Dynamics')
        plt.xlabel('Simulation Day')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True)
        
        # Food sources subplot
        plt.subplot(1, 2, 2)
        plt.plot(stats["step"], stats["grass_grown"], 'g-', label='Krill/Fish Availability')
        plt.title('Food Source Availability')
        plt.xlabel('Simulation Day')
        plt.ylabel('Available Food Patches')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(temp_dir, "antarctica_simulation.png")
        plt.savefig(viz_path)
        
        # Convert to base64
        with open(viz_path, "rb") as f:
            visualization = base64.b64encode(f.read()).decode('utf-8')
        
        return visualization
    
    async def run_simulation(self, model_type: str, config_params: Dict[str, Any], 
                          steps: int, log_callback: Optional[Callable] = None) -> Tuple[Dict[str, List[int]], str, List[str]]:
        """Run a simulation with the given parameters and return results and visualization."""
        if model_type not in self.config_paths:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Read configuration
        config_path = self.config_paths[model_type]
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
        
        config = read_config(config_path)
        
        # Override config with params
        config_overrides = config_params.get("config_params", {})
        for key, value in config_overrides.items():
            if key in config["simulation_metadata"]:
                config["simulation_metadata"][key] = value
        
        # Always ensure we're using larger population sizes
        if "num_predators" not in config_overrides:
            config["simulation_metadata"]["num_predators"] = 1000  # Leopard seals
        if "num_prey" not in config_overrides:
            config["simulation_metadata"]["num_prey"] = 9000      # Emperor penguins
        
        # Set up registry and runner
        registry = self.setup_registry()
        
        # Create runner
        try:
            runner = VectorizedRunner(config, registry)
            runner.init()
        except Exception as e:
            print(f"Error initializing runner: {e}")
            raise
        
        # Create temporary directory for visualization
        temp_dir = tempfile.mkdtemp()
        
        # Set up visualization if available
        try:
            visual = Plot(config["simulation_metadata"]["max_x"], config["simulation_metadata"]["max_y"])
        except Exception as e:
            print(f"Error setting up visualization: {e}")
            visual = None
        
        # Create list to store logs
        execution_logs = []
        if log_callback is None:
            log_callback = lambda step, substep, agent_type, action: execution_logs.append(
                f"Step {step}, Substep {substep}: {agent_type} {action}"
            )
        
        # Run simulation and capture state
        stats = {
            "step": [],
            "predators_alive": [],
            "prey_alive": [],
            "grass_grown": []
        }
        
        prev_dead_penguins = 0
        
        # Run simulation with logging
        for step in range(steps):
            for substep_idx in range(config["simulation_metadata"]["num_substeps_per_step"]):
                current_substep = str(substep_idx)
                
                # Log substep start
                substep_name = config["substeps"][current_substep]["name"]
                log_callback(step, substep_name, "system", "starting")
                
                # Before step
                runner.state["current_substep"] = current_substep
                
                # Run step
                runner.step(1)
                
                # After step - log results
                if substep_name == "Move":
                    # Log movement stats
                    penguins_alive = (runner.state['agents']['prey']['energy'] > 0).sum().item()
                    seals_alive = (runner.state['agents']['predator']['energy'] > 0).sum().item()
                    log_callback(step, substep_name, "penguins", f"moved, {penguins_alive} alive")
                    log_callback(step, substep_name, "seals", f"moved, {seals_alive} alive")
                elif substep_name == "Eat":
                    food_patches = (runner.state['objects']['grass']['growth_stage'] == 1).sum().item()
                    log_callback(step, substep_name, "penguins", f"consumed food, {food_patches} food patches remaining")
                elif substep_name == "Hunt":
                    penguins_caught = ((runner.state['agents']['prey']['energy'] == 0).sum().item() - prev_dead_penguins)
                    prev_dead_penguins = (runner.state['agents']['prey']['energy'] == 0).sum().item()
                    log_callback(step, substep_name, "seals", f"caught {penguins_caught} penguins")
                elif substep_name == "Grow":
                    food_grown = (runner.state['objects']['grass']['growth_stage'] == 1).sum().item()
                    log_callback(step, substep_name, "environment", f"grew food, now {food_grown} food patches available")
            
            # Collect statistics for this step
            penguins_alive = (runner.state['agents']['prey']['energy'] > 0).sum().item()
            seals_alive = (runner.state['agents']['predator']['energy'] > 0).sum().item()
            food_patches = (runner.state['objects']['grass']['growth_stage'] == 1).sum().item()
            
            stats["step"].append(step)
            stats["predators_alive"].append(seals_alive)
            stats["prey_alive"].append(penguins_alive)
            stats["grass_grown"].append(food_patches)
            
            # Capture visualization frame
            if visual:
                visual.capture(step, runner.state)
        
        # Create animation if visualization is available
        visualization = ""
        try:
            if visual:
                visual.compile(0)
                
                # Convert visualization to base64
                gif_path = os.path.join(temp_dir, "predator-prey-0.gif")
                if os.path.exists(gif_path):
                    with open(gif_path, "rb") as f:
                        visualization = base64.b64encode(f.read()).decode('utf-8')
                else:
                    # If no GIF was created, use our custom visualization
                    visualization = self.create_custom_visualization(stats, temp_dir)
            else:
                # Create custom visualization if Plot is not available
                visualization = self.create_custom_visualization(stats, temp_dir)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            # Create a simpler visualization as fallback
            try:
                visualization = self.create_custom_visualization(stats, temp_dir)
            except Exception as e2:
                print(f"Error creating fallback visualization: {e2}")
        
        return stats, visualization, execution_logs
    
    async def register_new_substep(self, substep_code: str, model_type: str) -> Tuple[bool, str]:
        """Register a new substep in the system."""
        try:
            # Create a temporary file for the substep
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, f"new_substep_{int(time.time())}.py")
            
            with open(temp_file_path, "w") as f:
                f.write(substep_code)
            
            # Import the module to register the substep
            module_name = os.path.basename(temp_file_path).replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find registered substeps in the module
            substep_classes = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and (
                    issubclass(obj, SubstepObservation) or 
                    issubclass(obj, SubstepAction) or 
                    issubclass(obj, SubstepTransition)
                ) and obj.__module__ == module.__name__:
                    substep_classes.append(name)
            
            return True, f"Successfully registered substeps: {', '.join(substep_classes)}"
        
        except Exception as e:
            return False, f"Error registering substep: {str(e)}"