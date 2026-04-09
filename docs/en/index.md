# CustomGrid Environment

A Gymnasium-based grid environment featuring an agent navigating a 4x5 grid to reach goal cells while avoiding a chasing ghost.

![Environment Preview](../Umgebung.png)

## Overview

CustomGrid is a turn-based environment where:
- An **agent** (robot with GPS) tries to reach one of the goal cells.
- A **ghost** chases the agent each turn.
- **Walls** block movement between certain cells.
- **Slip probability** adds stochasticity – the agent may slip perpendicular to the intended direction.
- **Coloured cells** provide visual information (red and green patterns).

## Quick Start

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent

# Create the interface
interface = AgentInterface(render=True, slip_probability=0.2)

# Reset and get initial observation
obs = interface.reset()

# Create your agent
agent = RandomPlayerAgent(interface.get_action_space())

# Run an episode
while not interface.is_terminated():
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)

# Get results
stats = interface.get_episode_stats()
print(f"Total reward: {stats['total_reward']}")

interface.close()
```

## Requirements

- Python 3.8+
- gymnasium
- numpy
- pygame
