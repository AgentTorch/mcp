# services/config_modifier.py
import os
import yaml
import logging
import copy
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config-modifier")

class SubstepTemplateManager:
    """Manages templates for different substep types"""
    
    @staticmethod
    def get_move_template():
        """Get template for Movement substep"""
        return {
            'name': 'Move',
            'description': 'Moving',
            'active_agents': ['predator', 'prey'],
            'observation': {
                'predator': {
                    'find_neighbors': {
                        'generator': 'FindNeighborsVmap',
                        'arguments': None,
                        'input_variables': {
                            'bounds': 'environment/bounds',
                            'adj_grid': 'network/agent_agent/predator_prey/adjacency_matrix',
                            'positions': 'agents/predator/coordinates'
                        },
                        'output_variables': ['possible_neighbors']
                    }
                },
                'prey': {
                    'find_neighbors': {
                        'generator': 'FindNeighborsVmap',
                        'arguments': None,
                        'input_variables': {
                            'bounds': 'environment/bounds',
                            'adj_grid': 'network/agent_agent/predator_prey/adjacency_matrix',
                            'positions': 'agents/prey/coordinates'
                        },
                        'output_variables': ['possible_neighbors']
                    }
                }
            },
            'policy': {
                'predator': {
                    'decide_movement': {
                        'generator': 'DecideMovementVmap',
                        'arguments': None,
                        'input_variables': {
                            'positions': 'agents/predator/coordinates',
                            'energy': 'agents/predator/energy'
                        },
                        'output_variables': ['next_positions']
                    }
                },
                'prey': {
                    'decide_movement': {
                        'generator': 'DecideMovementVmap',
                        'arguments': None,
                        'input_variables': {
                            'positions': 'agents/prey/coordinates',
                            'energy': 'agents/prey/energy'
                        },
                        'output_variables': ['next_positions']
                    }
                }
            },
            'transition': {
                'update_positions': {
                    'generator': 'UpdatePositionsVmap',
                    'arguments': None,
                    'input_variables': {
                        'prey_pos': 'agents/prey/coordinates',
                        'prey_energy': 'agents/prey/energy',
                        'pred_pos': 'agents/predator/coordinates',
                        'pred_energy': 'agents/predator/energy',
                        'prey_work': 'agents/prey/stride_work',
                        'pred_work': 'agents/predator/stride_work'
                    },
                    'output_variables': [
                        'prey_pos',
                        'prey_energy',
                        'pred_pos',
                        'pred_energy'
                    ]
                }
            }
        }
    
    @staticmethod
    def get_eat_template():
        """Get template for Eating substep"""
        return {
            'name': 'Eat',
            'description': 'Eating Grass',
            'active_agents': ['prey'],
            'observation': {
                'prey': None
            },
            'policy': {
                'prey': {
                    'find_eatable_grass': {
                        'generator': 'FindEatableGrassVmap',
                        'arguments': None,
                        'input_variables': {
                            'bounds': 'environment/bounds',
                            'positions': 'agents/prey/coordinates',
                            'grass_growth': 'objects/grass/growth_stage'
                        },
                        'output_variables': ['eatable_grass_positions']
                    }
                }
            },
            'transition': {
                'eat_grass': {
                    'generator': 'EatGrassVmap',
                    'arguments': None,
                    'input_variables': {
                        'energy': 'agents/prey/energy',
                        'grass_growth': 'objects/grass/growth_stage',
                        'growth_countdown': 'objects/grass/growth_countdown',
                        'bounds': 'environment/bounds',
                        'prey_pos': 'agents/prey/coordinates',
                        'nutrition': 'objects/grass/nutritional_value',
                        'regrowth_time': 'objects/grass/regrowth_time'
                    },
                    'output_variables': [
                        'energy',
                        'grass_growth',
                        'growth_countdown'
                    ]
                }
            }
        }
        
    @staticmethod
    def get_hunt_template():
        """Get template for Hunting substep"""
        return {
            'name': 'Hunt',
            'description': 'Hunting Prey',
            'active_agents': ['predator'],
            'observation': {
                'predator': None
            },
            'policy': {
                'predator': {
                    'find_targets': {
                        'generator': 'FindTargetsVmap',
                        'arguments': None,
                        'input_variables': {
                            'prey_pos': 'agents/prey/coordinates',
                            'pred_pos': 'agents/predator/coordinates'
                        },
                        'output_variables': ['target_positions']
                    }
                }
            },
            'transition': {
                'hunt_prey': {
                    'generator': 'HuntPreyVmap',
                    'arguments': None,
                    'input_variables': {
                        'prey_energy': 'agents/prey/energy',
                        'pred_energy': 'agents/predator/energy',
                        'nutritional_value': 'agents/prey/nutritional_value',
                        'prey_pos': 'agents/prey/coordinates',
                        'pred_pos': 'agents/predator/coordinates'
                    },
                    'output_variables': [
                        'prey_energy',
                        'pred_energy'
                    ]
                }
            }
        }
        
    @staticmethod
    def get_grow_template():
        """Get template for Growing substep"""
        return {
            'name': 'Grow',
            'description': 'Grow Grass',
            'active_agents': ['prey'],
            'observation': {'prey': None},
            'policy': {'prey': None},
            'transition': {
                'grow_grass': {
                    'generator': 'GrowGrassVmap',
                    'arguments': None,
                    'input_variables': {
                        'grass_growth': 'objects/grass/growth_stage',
                        'growth_countdown': 'objects/grass/growth_countdown'
                    },
                    'output_variables': [
                        'grass_growth',
                        'growth_countdown'
                    ]
                }
            }
        }
        
    @staticmethod
    def get_template_by_name(name):
        """Get a substep template by name"""
        name_lower = name.lower()
        if 'move' in name_lower:
            return SubstepTemplateManager.get_move_template()
        elif 'eat' in name_lower:
            return SubstepTemplateManager.get_eat_template()
        elif 'hunt' in name_lower:
            return SubstepTemplateManager.get_hunt_template()
        elif 'grow' in name_lower:
            return SubstepTemplateManager.get_grow_template()
        else:
            # Default to move template
            return SubstepTemplateManager.get_move_template()

class ConfigModifier:
    def __init__(self, config_path: str = None):
        """
        Initialize the ConfigModifier with the path to the config file.
        If path is not provided, it will look for config.yaml in the root directory.
        """
        # Find config file path
        if not config_path:
            config_path = "config.yaml"
            
        self.config_path = os.path.abspath(config_path)
        logger.info(f"Using config file at: {self.config_path}")
        
        # Load the base template from the provided config.yaml
        self.base_template = self.load_template_config()
        
        # Load the current working config
        self.config = self.load_config()
    
    def load_template_config(self) -> Dict[str, Any]:
        """Load the template config from the base file."""
        try:
            # Get the base directory where the current script is located
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(base_dir, "config.yaml")
            
            if os.path.exists(template_path):
                with open(template_path, 'r') as file:
                    template = yaml.safe_load(file)
                    logger.info(f"Loaded template config from {template_path}")
                    return template
            
            # If template file doesn't exist, create a minimal one
            logger.warning(f"Template config not found at {template_path}, using minimal template")
            return {}
        except Exception as e:
            logger.error(f"Error loading template config: {e}")
            return {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load the current config from file, or use the base template if not found."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    logger.info(f"Loaded config with {len(config.get('substeps', {}))} substeps")
                    return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using base template")
                return copy.deepcopy(self.base_template)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return copy.deepcopy(self.base_template)
    
    def save_config(self) -> bool:
        """Save the config back to the file."""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logger.info(f"Saved config to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def add_substep(self, index: str, substep_config: Dict[str, Any]) -> bool:
        """
        Add a substep to the config.
        
        Args:
            index: The index/key of the substep (e.g., '3')
            substep_config: The configuration for the substep
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Make sure substeps section exists
            if 'substeps' not in self.config:
                self.config['substeps'] = {}
            
            # Process substep config to ensure proper Vmap functions
            self._ensure_vmap_functions(substep_config)
            
            # Add the substep
            self.config['substeps'][index] = substep_config
            logger.info(f"Added substep '{substep_config.get('name', 'Unknown')}' with index {index}")
            
            # Save the config
            return self.save_config()
        except Exception as e:
            logger.error(f"Error adding substep: {e}")
            return False
    
    def _ensure_vmap_functions(self, substep_config: Dict[str, Any]):
        """Ensure that all functions in the substep config use Vmap versions."""
        # Function to recursively check and update generator names
        def update_generators(obj):
            if not isinstance(obj, dict):
                return
                
            if 'generator' in obj and isinstance(obj['generator'], str):
                # Check if the generator name needs 'Vmap' suffix
                gen_name = obj['generator']
                if not gen_name.endswith('Vmap') and not gen_name.startswith('read_'):
                    obj['generator'] = f"{gen_name}Vmap"
                    logger.info(f"Updated generator from {gen_name} to {obj['generator']}")
            
            # Recursively process nested dictionaries
            for key, value in obj.items():
                if isinstance(value, dict):
                    update_generators(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            update_generators(item)
        
        # Apply the updates
        update_generators(substep_config)
    
    def reset_to_template(self) -> bool:
        """Reset the config to the base template."""
        try:
            self.config = copy.deepcopy(self.base_template)
            logger.info("Reset config to base template")
            return self.save_config()
        except Exception as e:
            logger.error(f"Error resetting config: {e}")
            return False
    
    def get_all_substeps(self) -> Dict[str, Any]:
        """Get all current substeps in the config."""
        return self.config.get('substeps', {})
    
    def get_next_substep_index(self) -> str:
        """Get the next available substep index."""
        if 'substeps' not in self.config:
            return '0'
        
        # Get all existing indices as integers
        indices = [int(idx) for idx in self.config['substeps'].keys() if idx.isdigit()]
        
        # Return the next index
        return str(max(indices) + 1) if indices else '0'
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        try:
            metadata = self.config.get('simulation_metadata', {})
            substeps = self.config.get('substeps', {})
            
            logger.info("Configuration Summary:")
            logger.info(f"- Device: {metadata.get('device', 'unknown')}")
            logger.info(f"- Episodes: {metadata.get('num_episodes', 'unknown')}")
            logger.info(f"- Steps per episode: {metadata.get('num_steps_per_episode', 'unknown')}")
            logger.info(f"- Predators: {metadata.get('num_predators', 'unknown')}")
            logger.info(f"- Prey: {metadata.get('num_prey', 'unknown')}")
            logger.info(f"- Grass patches: {metadata.get('num_grass', 'unknown')}")
            logger.info(f"- Substeps: {len(substeps)}")
            
            for idx, substep in substeps.items():
                logger.info(f"  - [{idx}] {substep.get('name', 'Unknown')}: {substep.get('description', 'No description')}")
        except Exception as e:
            logger.error(f"Error printing config summary: {e}")
    
    def get_config_metrics(self) -> Dict[str, Any]:
        """Get basic metrics from the config."""
        metrics = {
            'num_predators': 0,
            'num_prey': 0,
            'num_substeps': 0,
            'has_required_files': False
        }
        
        try:
            if 'simulation_metadata' in self.config:
                metadata = self.config['simulation_metadata']
                metrics['num_predators'] = metadata.get('num_predators', 0)
                metrics['num_prey'] = metadata.get('num_prey', 0)
                metrics['num_substeps'] = len(self.config.get('substeps', {}))
                
                # Check if required file paths exist
                required_fields = [
                    "predator_coords_file",
                    "prey_coords_file",
                    "grass_coords_file",
                    "grass_growth_stage_file",
                    "grass_growth_countdown_file"
                ]
                
                metrics['has_required_files'] = all(field in metadata for field in required_fields)
        except Exception as e:
            logger.error(f"Error getting config metrics: {e}")
            
        return metrics
        
    def detect_substeps(self, message: str) -> List[str]:
        """
        Detect all substep types mentioned in a message.
        
        Args:
            message: User's message
            
        Returns:
            List[str]: List of detected substep types ('move', 'eat', 'hunt', 'grow')
        """
        message_lower = message.lower()
        detected = []
        
        # Simple keyword matching for each substep type
        move_keywords = ['move', 'movement', 'locomotion', 'travel', 'walk', 'swim', 'navigation', 'motion']
        eat_keywords = ['eat', 'feed', 'forage', 'food', 'consume', 'nutrition', 'grazing', 'krill', 'algae']
        hunt_keywords = ['hunt', 'chase', 'catch', 'predation', 'predator', 'capture', 'attack', 'pursue', 'seal', 'hunting']
        grow_keywords = ['grow', 'regrow', 'grass', 'vegetation', 'regrowth', 'growth', 'plant']
        
        # Check for each type
        if any(kw in message_lower for kw in move_keywords):
            detected.append('move')
        if any(kw in message_lower for kw in eat_keywords):
            detected.append('eat')
        if any(kw in message_lower for kw in hunt_keywords):
            detected.append('hunt')
        if any(kw in message_lower for kw in grow_keywords):
            detected.append('grow')
            
        return detected

    def add_move_substep(self) -> bool:
        """Add the standard Move substep (index 0) to the config."""
        return self.add_substep('0', SubstepTemplateManager.get_move_template())

    def add_eat_substep(self) -> bool:
        """Add the standard Eat substep (index 1) to the config."""
        return self.add_substep('1', SubstepTemplateManager.get_eat_template())

    def add_hunt_substep(self) -> bool:
        """Add the standard Hunt substep (index 2) to the config."""
        return self.add_substep('2', SubstepTemplateManager.get_hunt_template())

    def add_grow_substep(self) -> bool:
        """Add the standard Grow substep (index 3) to the config."""
        return self.add_substep('3', SubstepTemplateManager.get_grow_template())