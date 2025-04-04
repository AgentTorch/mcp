# test_simulation.py
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from services.simulation import SimulationService
from agent_torch.core.helpers import read_config, read_from_file, grid_network

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Create simulation service with optional vectorized mode
    use_vectorized = True  # Try vectorized version first
    
    service = SimulationService(use_vectorized=use_vectorized)
    
    # Run simulation
    print(f"Running {'vectorized ' if use_vectorized else ''}predator-prey simulation...")
    stats, logs = await service.run_simulation()
    
    # Display results
    if isinstance(stats, dict) and "error" in stats:
        print(f"Error: {stats['error']}")
        print("Trying again with standard (non-vectorized) implementation...")
        
        # Fallback to standard implementation
        service = SimulationService(use_vectorized=False)
        stats, logs = await service.run_simulation()
        
        if isinstance(stats, dict) and "error" in stats:
            print(f"Error with standard implementation: {stats['error']}")
            return
    
    # Log results
    print("\nSimulation Logs:")
    for log in logs:
        print(log)
    
    print("\nFinal Statistics:")
    print(f"Initial predators: {stats['predators_alive'][0]}")
    print(f"Final predators: {stats['predators_alive'][-1]}")
    print(f"Initial prey: {stats['prey_alive'][0]}")
    print(f"Final prey: {stats['prey_alive'][-1]}")
    print(f"Initial grass patches: {stats['grass_grown'][0]}")
    print(f"Final grass patches: {stats['grass_grown'][-1]}")
    
    # Plot results
    plot_results(stats)

def plot_results(stats):
    """Plot simulation results."""
    # Create output directory if it doesn't exist
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot population counts
    episodes = np.array(stats["episode"])
    steps = np.array(stats["step"])
    total_steps = len(stats["step"])
    step_numbers = np.arange(total_steps)
    
    # Population over time
    axs[0].plot(step_numbers, stats["predators_alive"], 'r-', label='Predators')
    axs[0].plot(step_numbers, stats["prey_alive"], 'b-', label='Prey')
    axs[0].set_title('Population Over Time')
    axs[0].set_xlabel('Simulation Step')
    axs[0].set_ylabel('Count')
    axs[0].legend()
    axs[0].grid(True)
    
    # Grass grown over time
    axs[1].plot(step_numbers, stats["grass_grown"], 'g-', label='Grown Grass')
    axs[1].set_title('Grass Growth Over Time')
    axs[1].set_xlabel('Simulation Step')
    axs[1].set_ylabel('Count')
    axs[1].legend()
    axs[1].grid(True)
    
    # Add overall title
    fig.suptitle('Predator-Prey Simulation Statistics', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'simulation_results.png'))
    
    print(f"\nPlot saved to {os.path.join(output_dir, 'simulation_results.png')}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())