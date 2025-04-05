Can you run a simulation of an Antarctic ecosystem with 1000 leopard seals (predators) and 9000 emperor penguins (prey) over 50 steps? I'm interested in understanding how the predator-prey dynamics evolve in this harsh environment, especially considering the limited food resources. Please analyze the population trends and explain any important ecological observations.

I'd also like to add a 'Grow' substep to the simulation that allows the grass to regrow over time. When a grass patch has been eaten by prey, its growth stage should be set to 0 and its growth countdown timer should start. Each step of the simulation should decrease this countdown, and when it reaches zero, the growth stage should be set back to 1, making it available for consumption again. This substep should use the GrowGrassVmap generator and take grass_growth and growth_countdown as input variables, with the same variables as outputs.

Please analyze the population trends and explain any important ecological observations.

  '3':
    name: 'Grow'
    description: 'Grow Grass'
    active_agents:
      - 'prey'
    observation:
      prey: null
    policy:
      prey: null
    transition:
      grow_grass:
        generator: 'GrowGrassVmap'
        arguments: null
        input_variables:
          grass_growth: 'objects/grass/growth_stage'
          growth_countdown: 'objects/grass/growth_countdown'
        output_variables:
          - grass_growth
          - growth_countdown