# CustomGrid Environment

A Gymnasium-based grid environment where an agent navigates a 4x5 grid to reach goal cells while avoiding a chasing ghost.

## Interactive Notebooks

You can try the CNN training and the environment directly in your browser using Google Colab:

- **CNN Training Tutorial**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb)
- **Environment Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb)
- **Interactive GUI**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb)

![Environment Preview](../Umgebung.png)

## Overview

CustomGrid is a turn-based environment where:
- An **agent** (robot with GPS) attempts to reach one of the target cells.
- A **ghost** chases the agent in every round.
- **Walls** block movement between certain cells.
- **Slip probability** provides stochasticity – the agent can slip perpendicular to the intended direction.
- **Colored cells** provide visual information (red and green patterns).

## Goal of the Environment

The primary goal of the environment is to develop a mobile agent that visits various locations (which can only be detected visually or acoustically) in minimal time and returns to the starting location without colliding with other road users (similar to the Traveling Salesman Problem).

### Given Resources
- **Agent**: Equipped with sensors, a chassis, and a small computer.
- **Sensors**: Camera, microphone, color sensor.
- **Map**: Information about which fields can perceive which optical and acoustic stimuli.

### Example Task
At the beginning, the agent receives an instruction like:
"Visit the following three fields in optimal order and return to the starting location:
- the field where you hear piano music,
- the field where you see the picture of the dog and hear rock music,
- the field with the text 'Goal'."

**Important**: The agent does not know its position at time $t=0$ and must estimate it using its sensors (localization).

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
print(f"Total Reward: {stats['total_reward']}")

interface.close()
```

## Requirements

- Python 3.8+
- gymnasium
- numpy
- pygame
