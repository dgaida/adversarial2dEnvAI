# Getting Started

Welcome to `custom_grid_env`! This page will help you get familiar with the environment quickly.

## Concepts

The environment simulates a grid where an agent must navigate. The following components play a central role:

1.  **AgentInterface**: The primary interface for your AI agents. It encapsulates the environment, the ghost, and the particle filter.  
2.  **Particle Filter**: A mechanism for estimating the agent's position if it is not exactly known (localization).  
3.  **CNN Classification**: A neural network that processes images of grid cells to recognize objects like dogs or flowers.  

## First Experiments


| Notebook | Description | Link |
| :--- | :--- | :--- |
| **Interactive GUI** | Full dashboard with sensors and PF visualization. | [![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb) |
| **Environment Demo** | Learn the basics of control. | [![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb) |
| **CNN Training** | Learn how to train the neural network that the agent uses for object recognition. | [![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb) |


## A Simple Example

Here is a minimal script to start an agent with random movements:

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent

# Initialize interface
interface = AgentInterface(render=True)
obs = interface.reset()

# Create agent
agent = RandomPlayerAgent(interface.get_action_space())

# Run episode
for _ in range(100):
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)
    if done:
        break

interface.close()
```

## Further Tutorials

Check out our detailed tutorials:  
- [CNN Training Tutorial](tutorial/cnn.md)  
- [Particle Filter Tutorial](tutorial/particle_filter.md)  
