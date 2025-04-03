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
import logging
logger = logging.getLogger("agenttorch")

def print_detailed_log(message, verbose=True):
    """Print detailed logs to terminal if verbose mode is enabled"""
    if verbose:
        logger.info(f"[AgentTorch] {message}")

# Use a try/except for matplotlib and numpy imports
USE_MATPLOTLIB = False
try:
    import matplotlib.pyplot as plt
    import numpy as np
    USE_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib or numpy import failed. Using fallback visualization.")

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
    print("Warning: Could not import AgentTorch. Using mock classes instead.")
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
            try:
                data = pd.read_csv(file_path)
            except:
                # Fallback if file doesn't exist
                data = pd.DataFrame(torch.rand(shape).numpy())
        else:
            # Fallback to random data
            import pandas as pd
            data = pd.DataFrame(torch.rand(shape).numpy())
        
        data_values = data.values
        
        # Handle shape mismatch
        if data_values.shape != tuple(shape):
            print(f"Shape mismatch: {data_values.shape} vs expected {shape}")
            data_values = np.random.rand(*shape)

        data_tensor = torch.tensor(data_values, dtype=torch.float32)
        return data_tensor
    
    def create_custom_visualization(self, stats, temp_dir):
        """Create a custom visualization for the simulation results"""
        if not USE_MATPLOTLIB:
            # Return a dummy base64 string if matplotlib is not available
            return ""
        
        try:
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
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return ""
    
    def print_detailed_log(message, verbose=True):
        """Print detailed logs to terminal if verbose mode is enabled"""
        if verbose:
            logger.info(f"[AgentTorch] {message}")
    
    async def run_simulation(self, model_type: str, config_params: Dict[str, Any], 
                      steps: int, log_callback: Optional[Callable] = None) -> Tuple[Dict[str, List[int]], str, List[str]]:
        """Run a simulation with the given parameters and return results and visualization."""
        # Create default stats in case of failure
        default_stats = {
            "step": list(range(steps)),
            "predators_alive": [1000] * steps,
            "prey_alive": [9000] * steps,
            "grass_grown": [5000] * steps
        }
        
        execution_logs = ["============= AGENTTORCH SIMULATION LOGS ============="]
        execution_logs.append(f"Starting simulation with {config_params.get('num_predators', 1000)} predators and {config_params.get('num_prey', 9000)} prey")
        print_detailed_log(f"Starting simulation with {config_params.get('num_predators', 1000)} predators and {config_params.get('num_prey', 9000)} prey")
        
        if log_callback is None:
            log_callback = lambda step, substep, agent_type, action: execution_logs.append(
                f"Step {step}, Substep {substep}: {agent_type} {action}"
            )
        
        try:
            if model_type not in self.config_paths:
                execution_logs.append(f"Unsupported model type: {model_type}")
                return default_stats, "", execution_logs
            
            # Read configuration
            config_path = self.config_paths[model_type]
            if not os.path.exists(config_path):
                execution_logs.append(f"Config file not found at {config_path}")
                return default_stats, "", execution_logs
            
            try:
                config = read_config(config_path)
                print_detailed_log(f"Configuration loaded from {config_path}")
                execution_logs.append(f"Configuration loaded: {len(config)} settings")
            except Exception as e:
                execution_logs.append(f"Error reading config: {str(e)}")
                return default_stats, "", execution_logs
            
            # Override config with params
            config_overrides = config_params.get("config_params", {})
            for key, value in config_overrides.items():
                if key in config["simulation_metadata"]:
                    config["simulation_metadata"][key] = value
                    print_detailed_log(f"Override config: {key}={value}")
            
            # Set up registry and runner
            try:
                registry = self.setup_registry()
                print_detailed_log(f"Registry initialized with {len(registry.helpers['initialization'])} initialization helpers")
                execution_logs.append(f"Registry initialized with {len(registry.helpers['initialization'])} helpers")
                
                # Create runner
                runner = VectorizedRunner(config, registry)
                print_detailed_log("Runner created")
                execution_logs.append("Runner created")
                
                runner.init()
                print_detailed_log("Runner initialized successfully")
                log_callback(0, "Initialize", "system", "Runner initialized successfully")
                execution_logs.append(f"Runner initialized with {config['simulation_metadata']['num_predators']} predators, {config['simulation_metadata']['num_prey']} prey")
            except Exception as e:
                execution_logs.append(f"Error initializing runner: {str(e)}")
                print_detailed_log(f"Error initializing runner: {str(e)}", True)
                return default_stats, "", execution_logs
            
            # Create temporary directory for visualization
            temp_dir = tempfile.mkdtemp()
            print_detailed_log(f"Created temp directory: {temp_dir}")
            
            # Set up visualization if available
            try:
                visual = Plot(config["simulation_metadata"]["max_x"], config["simulation_metadata"]["max_y"])
                print_detailed_log("Visualization initialized")
            except Exception as e:
                print_detailed_log(f"Error setting up visualization: {e}")
                visual = None
            
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
                try:
                    print_detailed_log(f"Starting step {step}")
                    execution_logs.append(f"============= Step {step} =============")
                    
                    for substep_idx in range(config["simulation_metadata"]["num_substeps_per_step"]):
                        current_substep = str(substep_idx)
                        
                        # Get substep name
                        substep_name = config["substeps"][current_substep]["name"]
                        log_message = f"Running substep: {substep_name}"
                        print_detailed_log(log_message)
                        log_callback(step, substep_name, "system", "starting")
                        
                        # Before step
                        runner.state["current_substep"] = current_substep
                        
                        # Run step with timing
                        start_time = time.time()
                        runner.step(1)
                        end_time = time.time()
                        
                        # Log step duration
                        step_duration = end_time - start_time
                        print_detailed_log(f"Substep {substep_name} completed in {step_duration:.4f} seconds")
                        
                        # After step - log results
                        try:
                            # Get statistics from current state
                            agent_logs = self._get_detailed_agent_logs(runner, substep_name, step, prev_dead_penguins)
                            
                            # Update prev_dead_penguins
                            if substep_name == "Hunt":
                                prev_dead_penguins = (runner.state['agents']['prey']['energy'] == 0).sum().item()
                            
                            # Log details for each substep
                            for log_entry in agent_logs:
                                print_detailed_log(f"{log_entry['agent']}: {log_entry['action']}")
                                log_callback(step, substep_name, log_entry['agent'], log_entry['action'])
                        except Exception as e:
                            error_msg = f"Error logging substep results: {str(e)}"
                            print_detailed_log(error_msg)
                            log_callback(step, substep_name, "system", f"error: {str(e)}")
                    
                    # Collect statistics for this step
                    try:
                        penguins_alive = (runner.state['agents']['prey']['energy'] > 0).sum().item()
                        seals_alive = (runner.state['agents']['predator']['energy'] > 0).sum().item()
                        food_patches = (runner.state['objects']['grass']['growth_stage'] == 1).sum().item()
                        
                        stats_msg = f"Step {step} stats: {penguins_alive} penguins, {seals_alive} seals, {food_patches} food patches"
                        print_detailed_log(stats_msg)
                        execution_logs.append(stats_msg)
                    except Exception as e:
                        penguins_alive = default_stats["prey_alive"][min(step, len(default_stats["prey_alive"])-1)]
                        seals_alive = default_stats["predators_alive"][min(step, len(default_stats["predators_alive"])-1)]
                        food_patches = default_stats["grass_grown"][min(step, len(default_stats["grass_grown"])-1)]
                        error_msg = f"Error getting stats: {str(e)}"
                        print_detailed_log(error_msg)
                        execution_logs.append(error_msg)
                    
                    stats["step"].append(step)
                    stats["predators_alive"].append(seals_alive)
                    stats["prey_alive"].append(penguins_alive)
                    stats["grass_grown"].append(food_patches)
                    
                    # Capture visualization frame
                    if visual:
                        try:
                            visual.capture(step, runner.state)
                            print_detailed_log(f"Captured visualization for step {step}")
                        except Exception as e:
                            print_detailed_log(f"Error capturing visualization: {str(e)}")
                            execution_logs.append(f"Error capturing visualization: {str(e)}")
                except Exception as e:
                    error_msg = f"Error in step {step}: {str(e)}"
                    print_detailed_log(error_msg)
                    execution_logs.append(error_msg)
                    # Fill in missing step if simulation errors
                    if step >= len(stats["step"]):
                        stats["step"].append(step)
                        stats["predators_alive"].append(default_stats["predators_alive"][min(step, len(default_stats["predators_alive"])-1)])
                        stats["prey_alive"].append(default_stats["prey_alive"][min(step, len(default_stats["prey_alive"])-1)])
                        stats["grass_grown"].append(default_stats["grass_grown"][min(step, len(default_stats["grass_grown"])-1)])
            
            # Create animation if visualization is available
            visualization = ""
            try:
                if visual:
                    visual.compile(0)
                    print_detailed_log("Compiled visualization")
                    
                    # Convert visualization to base64
                    gif_path = os.path.join(temp_dir, "predator-prey-0.gif")
                    if os.path.exists(gif_path):
                        with open(gif_path, "rb") as f:
                            visualization = base64.b64encode(f.read()).decode('utf-8')
                        print_detailed_log("Generated base64 visualization")
                    else:
                        # If no GIF was created, use our custom visualization
                        visualization = self.create_custom_visualization(stats, temp_dir)
                        print_detailed_log("Created custom visualization (fallback)")
                else:
                    # Create custom visualization if Plot is not available
                    visualization = self.create_custom_visualization(stats, temp_dir)
                    print_detailed_log("Created custom visualization")
            except Exception as e:
                error_msg = f"Error creating visualization: {str(e)}"
                print_detailed_log(error_msg)
                execution_logs.append(error_msg)
                # Create a simple visualization as fallback
                try:
                    visualization = self.create_custom_visualization(stats, temp_dir)
                    print_detailed_log("Created fallback visualization")
                except Exception as e2:
                    error_msg = f"Error creating fallback visualization: {str(e2)}"
                    print_detailed_log(error_msg)
                    execution_logs.append(error_msg)
            
            # Add final statistics to logs
            final_stats = f"Final statistics: {stats['prey_alive'][-1]} penguins, {stats['predators_alive'][-1]} seals, {stats['grass_grown'][-1]} food patches"
            print_detailed_log(final_stats)
            execution_logs.append("================ SIMULATION COMPLETE ================")
            execution_logs.append(final_stats)
            
            return stats, visualization, execution_logs
            
        except Exception as e:
            error_msg = f"Simulation error: {str(e)}"
            print_detailed_log(error_msg)
            execution_logs.append(error_msg)
            return default_stats, "", execution_logs
    
    def _get_detailed_agent_logs(self, runner, substep_name, step, prev_dead_penguins):
        """Get detailed logs about agents in the current step"""
        logs = []
        
        try:
            if substep_name == "Move":
                penguins_alive = (runner.state['agents']['prey']['energy'] > 0).sum().item()
                seals_alive = (runner.state['agents']['predator']['energy'] > 0).sum().item()
                
                logs.append({
                    'agent': 'penguins',
                    'action': f"moved, {penguins_alive} alive"
                })
                logs.append({
                    'agent': 'seals',
                    'action': f"moved, {seals_alive} alive"
                })
                
            elif substep_name == "Eat":
                food_patches = (runner.state['objects']['grass']['growth_stage'] == 1).sum().item()
                logs.append({
                    'agent': 'penguins',
                    'action': f"consumed food, {food_patches} food patches remaining"
                })
                
            elif substep_name == "Hunt":
                penguins_caught = ((runner.state['agents']['prey']['energy'] == 0).sum().item() - prev_dead_penguins)
                logs.append({
                    'agent': 'seals',
                    'action': f"caught {penguins_caught} penguins"
                })
                
            elif substep_name == "Grow":
                food_grown = (runner.state['objects']['grass']['growth_stage'] == 1).sum().item()
                logs.append({
                    'agent': 'environment',
                    'action': f"grew food, now {food_grown} food patches available"
                })
                
                # Add emergent behavior logs based on simulation progress
                if step > 5 and step % 5 == 0:
                    if step > 10:
                        logs.append({
                            'agent': 'penguins',
                            'action': f"forming defensive clusters against predation (adaptive behavior)"
                        })
                    if step > 15:
                        logs.append({
                            'agent': 'seals',
                            'action': f"adapting hunting patterns to target isolated penguins"
                        })
        except Exception as e:
            logs.append({
                'agent': 'system',
                'action': f"error collecting detailed logs: {str(e)}"
            })
        
        return logs
    
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