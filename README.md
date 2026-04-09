# CustomGrid Environment

[![Tests](https://github.com/user/custom_grid_env/actions/workflows/tests.yml/badge.svg)](https://github.com/user/custom_grid_env/actions/workflows/tests.yml)
[![Version](https://github.com/user/custom_grid_env/actions/workflows/version.yml/badge.svg)](https://github.com/user/custom_grid_env/actions/workflows/version.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.1-blue)
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

## Documentation

See the [docs](docs/README.md) directory for detailed information:
- [Environment](docs/environment.md)
- [Gameplay](docs/gameplay.md)
- [Observations](docs/observations.md)
- [Rewards](docs/rewards.md)
- [API Reference](docs/api.md)

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
