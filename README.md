# CustomGrid Environment 🤖👻

![CustomGrid Environment](docs/Umgebung.png)
[![Version](https://img.shields.io/github/v/tag/dgaida/adversarial2dEnvAI?label=version)](https://github.com/dgaida/adversarial2dEnvAI/tags)
[![Code Quality](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/lint.yml)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/adversarial2dEnvAI/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/adversarial2dEnvAI)
[![codecov](https://codecov.io/gh/dgaida/adversarial2dEnvAI/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/adversarial2dEnvAI)

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/adversarial2dEnvAI/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![interrogate](docs/assets/interrogate.svg)](https://dgaida.github.io/adversarial2dEnvAI/dev/metrics/)
[![Tests](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/tests.yml)

An advanced Gymnasium-based grid environment for Reinforcement Learning and Robotics tutorials, used in the **AI lecture at TH Köln**. CustomGrid features an agent navigating a stochastic environment with imperfect sensors, adversarial elements, and complex state estimation.

---

## 🎯 Goal of the Environment

The primary goal of this environment is to teach students how to develop an autonomous agent capable of achieving complex tasks defined by a user. These tasks may include visiting specific cells identified by visual or acoustic stimuli in an optimal order and returning to the starting position, often beginning from an unknown location.

A key challenge is the integration of multiple modules—such as **Speech-to-Text** for task understanding, **Vision/Acoustic sensors** for identification, and **Bayesian filters** for localization—to allow the agent to solve high-level goals (e.g., "Visit the dog, then the goal").

---

## 🌟 Key Features

*   **Turn-Based Adversarial Gameplay**: Compete against a ghost in a 4x5 grid.  
*   **Adversarial Search**: Integrated **Minimax** and **Expectimax** agents for strategic planning.  
*   **Stochastic Movement**: Realistic physics with *Perpendicular* and *Longitudinal* slipping.  
*   **Imperfect Perception**:  
    *   **Noisy Color Sensor**: Ground color detection with 80% accuracy.  
    *   **CNN-Based Vision**: Real-time item classification using a trained CNN.  
*   **State Estimation**: Integrated **Particle Filter** for Bayesian localization.  
*   **Interactive Visualization**:  
    *   Rich Pygame-based renderer.  
    *   Interactive Google Colab GUI with real-time 2D probability distribution.  
*   **Customizable Ghost AI**: Switch between shortest-path chasing, random movement, and minimax.  

---

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/dgaida/adversarial2dEnvAI.git
```

### Anaconda Environment

You can also use Anaconda to manage your environment:

```bash
# Create and activate the environment from the provided file
conda env create -f environment.yml
conda activate custom_grid_env

# Install the package in editable mode
pip install -e .
```

### Basic Usage

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.adversarial_agents import MinimaxAgent

# Initialize the interface with Particle Filter and rendering
interface = AgentInterface(render=True, slip_probability=0.2)
obs = interface.reset()

# Use a strategic Minimax agent
agent = MinimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=4)

while not interface.is_terminated():
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)

    # Access estimated position from Particle Filter
    est_pos = info['estimated_pos']['cell_pos']
    print(f"Estimated Position: {est_pos}")

interface.close()
```

---

## 📓 Interactive Notebooks

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **Interactive GUI** | Full dashboard with sensors and PF visualization. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb) |
| **Environment Demo** | Basic programmatic interaction and API walkthrough. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb) |
| **CNN Training** | Tutorial on training the vision model. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb) |

---

## 📖 Documentation

Visit our [Documentation Site](https://dgaida.github.io/adversarial2dEnvAI/) for:  
*   🇩🇪 [Deutsch](https://dgaida.github.io/adversarial2dEnvAI/dev/)  
*   🇺🇸 [English](https://dgaida.github.io/adversarial2dEnvAI/dev/en/)  

Includes tutorials on [Adversarial Search](https://dgaida.github.io/adversarial2dEnvAI/dev/tutorial/adversarial_search/), [Localization](https://dgaida.github.io/adversarial2dEnvAI/dev/usage/localization/), and more.

---

## 🛠 Development

### Setup

```bash
git clone https://github.com/dgaida/adversarial2dEnvAI.git
cd adversarial2dEnvAI
pip install -e .[dev]
```

### Running Tests

```bash
PYTHONPATH=src python3 -m pytest
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgment

This repository is a fork of [Praxisprojekt](https://github.com/Malte-18/Praxisprojekt) by M.B., Praxisprojekt, TH Köln, 2026.
