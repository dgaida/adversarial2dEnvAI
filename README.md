# CustomGrid Environment

[![Tests](https://github.com/user/custom_grid_env/actions/workflows/tests.yml/badge.svg)](https://github.com/user/custom_grid_env/actions/workflows/tests.yml)
[![Version](https://github.com/user/custom_grid_env/actions/workflows/version.yml/badge.svg)](https://github.com/user/custom_grid_env/actions/workflows/version.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/github/v/tag/dgaida/adversarial2dEnvAI?label=version)](https://github.com/dgaida/adversarial2dEnvAI/tags)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Quality](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/adversarial2dEnvAI/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/adversarial2dEnvAI/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/adversarial2dEnvAI)

A Gymnasium-based grid environment featuring an agent navigating a 4x5 grid to reach goal cells while avoiding a chasing ghost.

## Overview

CustomGrid is a turn-based environment where:
- An **agent** (robot with GPS) tries to reach one of the goal cells
- A **ghost** chases the agent each turn
- **Walls** block movement between certain cells
- **Slip probability** adds stochasticity - the agent may slip perpendicular to intended direction
- **Coloured cells** provide visual information (red and green patterns)

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

## CNN-based Item Classification

The environment includes a Convolutional Neural Network (CNN) that automatically classifies items (dogs and flowers) when the agent stands on them. This feature demonstrates how neural networks can be integrated into reinforcement learning environments for perception tasks.

- **Data Generation**: Images are procedurally generated with different backgrounds.
- **Model**: A simple CNN built with TensorFlow/Keras.
- **Integration**: The `PygameRenderer` uses the trained model to provide real-time predictions in the info panel.

See the [CNN Tutorial](docs/en/tutorial/cnn.md) (or [German version](docs/de/tutorial/cnn.md)) for more details.

### Interactive Notebooks

You can try out the CNN training and the environment directly in your browser using Google Colab:

- **CNN Training Tutorial**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb)
- **Environment Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb)
- **Interactive GUI**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb)

## Documentation

Detailed documentation is available in the `docs/` directory or online at [dgaida.github.io/adversarial2dEnvAI](https://dgaida.github.io/adversarial2dEnvAI/).

## Requirements

- Python 3.8+
- gymnasium
- numpy
- pygame

## Installation

```bash
pip install .
```

## Running the Demo

You can now use the package as shown in the Quick Start.
