# CustomGrid Environment 🤖👻

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/adversarial2dEnvAI/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/adversarial2dEnvAI/actions/workflows/tests.yml)

An advanced Gymnasium-based grid environment for Reinforcement Learning and Robotics tutorials. CustomGrid features an agent navigating a stochastic environment with imperfect sensors, adversarial elements, and complex state estimation.

---

## 🌟 Key Features

*   **Turn-Based Adversarial Gameplay**: An agent competes against a ghost in a 4x5 grid.
*   **Stochastic Movement**: Support for *Perpendicular* and *Longitudinal* slipping probabilities.
*   **Imperfect Perception**:
    *   **Noisy Color Sensor**: Ground color detection with 80% accuracy.
    *   **CNN-Based Vision**: Real-time item classification (dogs, flowers, background) using a trained Convolutional Neural Network.
*   **State Estimation**: Integrated **Particle Filter** for Bayesian localization, combining vision and color sensor data.
*   **Interactive Visualization**:
    *   Rich Pygame-based renderer.
    *   Interactive Google Colab GUI with real-time 2D probability distribution (contour plots).
*   **Customizable Ghost AI**: Switch between shortest-path chasing and random movement.

---

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/dgaida/adversarial2dEnvAI.git
```

### Basic Usage

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent

# Initialize the interface with Particle Filter and rendering
interface = AgentInterface(render=True, slip_probability=0.2)
obs = interface.reset()

agent = RandomPlayerAgent(interface.get_action_space())

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

Experience the environment directly in your browser:

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **Interactive GUI** | Full dashboard with sensors, PF visualization, and manual control. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb) |
| **Environment Demo** | Basic programmatic interaction and API walkthrough. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb) |
| **CNN Training** | Step-by-step tutorial on training the vision model. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb) |

---

## 📖 Documentation

For detailed guides on localization, rewards, and environment configuration, visit our [Documentation Site](https://dgaida.github.io/adversarial2dEnvAI/).

Available in:
*   🇩🇪 [Deutsch](https://dgaida.github.io/adversarial2dEnvAI/de/)
*   🇺🇸 [English](https://dgaida.github.io/adversarial2dEnvAI/en/)

---

## 🛠 Development

### Setup

```bash
git clone https://github.com/dgaida/adversarial2dEnvAI.git
cd adversarial2dEnvAI
pip install -e .
```

### Running Tests

```bash
PYTHONPATH=src python3 -m pytest
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
