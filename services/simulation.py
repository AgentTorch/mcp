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
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agenttorch")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

def print_detailed_log(message, verbose=True):
    """Print detailed logs to terminal if verbose mode is enabled"""
    if verbose:
        logger.info(f"[AgentTorch] {message}")

# Import AgentTorch core components
try:
    from agent_torch.core import Registry, Runner, VectorizedRunner
    from agent_torch.core.helpers import read_config, grid_network, read_from_file
    
    # Import predator-prey substeps
    from agent_torch.models.predator_prey.substeps import *
    
    # Import vectorized substeps if available
    try:
        from agent_torch.models.predator_prey.vmap_substeps import *
        logger.info("Successfully imported vmap_substeps")
        USE_VECTORIZED = True
    except ImportError:
        logger.warning("vmap_substeps not available, using standard substeps")
        USE_VECTORIZED = False
    
    # Import helper functions
    from agent_torch.models.predator_prey.helpers.random import random_float, random_int
    from agent_torch.models.predator_prey.helpers.map import map_network
    
    AGENTTORCH_AVAILABLE = True
except ImportError as e:
    logger.error(f"AgentTorch import error: {e}")
    AGENTTORCH_AVAILABLE = False
    USE_VECTORIZED = False

class SimulationService:
    def __init__(self, use_vectorized=False):
        # Set up root directory
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.use_vectorized = use_vectorized and USE_VECTORIZED
        
        # Create minimal config file if it doesn't exist
        self.create_minimal_config()
    
    def create_minimal_config(self):
        """Create a minimal configuration file for predator-prey simulation."""
        config_path = os.path.join(self.root_dir, "config.yaml")
    
    def setup_registry(self):
        """Set up the registry with necessary components."""
        registry = Registry()
        
        # Register essential functions
        registry.register(custom_read_from_file, "read_from_file", "initialization")
        registry.register(grid_network, "grid", key="network")
        registry.register(map_network, "map", key="network")
        registry.register(random_float, "random_float", "initialization")
        registry.register(random_int, "random_int", "initialization")
        
        logger.info("Registry set up with required functions")
        return registry
    
    def safe_count(self, tensor):
        """Safely count elements in a tensor, handling multi-element tensors properly."""
        if tensor.numel() == 0:
            return 0
            
        count = tensor.sum()
        return int(count.detach().cpu().numpy()) if count.numel() > 1 else int(count.item())
    
    async def run_simulation(self, reload_config=True):
        """
        Run a basic predator-prey simulation.
        
        Args:
            reload_config: Whether to reload the config file before running.
                           Set to True to pick up any config changes.
        """
        if not AGENTTORCH_AVAILABLE:
            return {"error": "AgentTorch not available"}, []
        
        config_path = os.path.join(self.root_dir, "config.yaml")
        if not os.path.exists(config_path):
            self.create_minimal_config()
        
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            # Always reload the config file to get the latest changes
            if reload_config:
                logger.info("Reloading configuration to pick up any changes")
            config = read_config(config_path, register_resolvers=False)
            
            # Set up registry and runner
            registry = self.setup_registry()
            
            # Choose runner based on availability of vectorized implementation
            RunnerClass = VectorizedRunner if self.use_vectorized else Runner
            runner = RunnerClass(config, registry)
            
            # Initialize the runner
            logger.info(f"Initializing {'vectorized ' if self.use_vectorized else ''}runner...")
            runner.init()
            
            # Get simulation parameters
            num_episodes = config['simulation_metadata']['num_episodes']
            num_steps_per_episode = config['simulation_metadata']['num_steps_per_episode']
            
            # Statistics to track
            stats = {
                "episode": [],
                "step": [],
                "predators_alive": [],
                "prey_alive": [],
                "grass_grown": []
            }
            
            # Run simulation steps
            logs = []
            logs.append(f"Starting {'vectorized ' if self.use_vectorized else ''}predator-prey simulation")
            
            # Run episodes
            for episode in range(num_episodes):
                logger.info(f"Starting episode {episode+1}/{num_episodes}")
                logs.append(f"Episode {episode+1}/{num_episodes}")
                
                # Reset runner for new episode
                runner.reset()
                
                # Run steps in each episode
                for step in range(num_steps_per_episode):
                    # Run one step
                    runner.step(1)
                    
                    # Get current state
                    current_state = runner.state
                    
                    # Calculate populations using safe counting method
                    pred_alive = self.safe_count(current_state['agents']['predator']['energy'] > 0)
                    prey_alive = self.safe_count(current_state['agents']['prey']['energy'] > 0)
                    grass_grown = self.safe_count(current_state['objects']['grass']['growth_stage'] == 1)
                    
                    # Store stats
                    stats["episode"].append(episode)
                    stats["step"].append(step)
                    stats["predators_alive"].append(pred_alive)
                    stats["prey_alive"].append(prey_alive)
                    stats["grass_grown"].append(grass_grown)
                    
                    # Log step
                    log_msg = f"Step {step+1}: {pred_alive} predators, {prey_alive} prey, {grass_grown} grass patches"
                    logs.append(log_msg)
                    logger.info(log_msg)
            
            logs.append("Simulation completed successfully")
            return stats, logs
        
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            return {"error": str(e)}, [f"Error: {str(e)}"]

    def print_detailed_log(message, verbose=True):
        """Print detailed logs to terminal if verbose mode is enabled"""
        if verbose:
            logger.info(f"[AgentTorch] {message}")


def custom_read_from_file(shape, params):
    """
    Custom file reader that handles relative paths correctly.
    """
    file_path = params["file_path"]
    
    # Make path absolute if needed
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_dir, file_path)
    
    print(f"Reading file: {file_path}")
    
    if file_path.endswith("csv"):
        data = pd.read_csv(file_path)
    
    data_values = data.values
    # assert data_values.shape == tuple(shape), f"Shape mismatch: {data_values.shape} vs expected {shape}"

    data_tensor = torch.tensor(data_values, dtype=torch.float32)
    return data_tensor