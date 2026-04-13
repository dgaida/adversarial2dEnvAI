# Agent Instructions for CustomGrid Environment

Welcome! This file contains specific instructions and tips for working with the `custom_grid_env` codebase.

## 🏗 Project Architecture

*   **`src/custom_grid_env/env.py`**: The core Gymnasium environment. Handles grid logic, movement, and collision.
*   **`src/custom_grid_env/interface.py`**: A high-level wrapper (`AgentInterface`) that manages turns (agent then ghost) and integrates the Particle Filter. **This is the primary entry point for users.**
*   **`src/custom_grid_env/renderer.py`**: Decoupled Pygame-based rendering.
*   **`src/custom_grid_env/sensors.py`**: Standalone CNN-based vision for item classification.
*   **`src/custom_grid_env/particle_filter.py`**: Implementation of the Particle Filter for localization.
*   **`src/custom_grid_env/colab_gui.py`**: Ipywidgets-based GUI specifically designed for Google Colab.

## 📏 Coding Standards

*   **Docstrings**: Use **Google-style** docstrings for all classes and methods.
*   **Formatting**: Use `black` for code formatting.
*   **Logging**: Use the centralized logger in `src/custom_grid_env/logger.py`.
    *   `INFO` level for user-facing status (e.g., estimated position).
    *   `DEBUG` level for internal state (e.g., raw particle positions) which is written to `custom_grid_env.log`.
*   **Type Hinting**: Use type hints for all function arguments and return values.

## 🧪 Testing

*   Always run tests using `PYTHONPATH=src python3 -m pytest` before submitting changes.
*   Unit tests are located in `tests/`.

## 🎨 Visualization

*   When modifying `ColabGUI`, ensure that any new plots or widgets are compatible with the headless environment (using the `dummy` SDL videodriver).
*   Probability distributions should be visualized as **contour plots** (Höhendiagramme) for clarity in 2D space.

## 🤖 Ghost Behaviors

*   New ghost behaviors should be implemented as classes inheriting from `custom_grid_env.agents.base_agent.BaseAgent`.
*   They can be swapped at runtime via `AgentInterface.set_ghost_agent(agent_class)`.
