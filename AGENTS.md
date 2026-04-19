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

## 📝 Documentation Guidelines

*   **List Formatting**: In all Markdown files (`docs/`), every list item (lines starting with `-`, `*`, or `1.`) **must** end with two spaces (`  `) to ensure proper vertical list formatting and line breaks in the rendered MkDocs output.  

## 🚀 GitHub Actions

The repository uses the following GitHub Actions for CI/CD:  
*   **Linting**: Runs `ruff` and `interrogate` (95% coverage requirement) to ensure code quality and documentation.  
*   **Tests**: Runs `pytest` and generates code coverage reports.  
*   **Documentation**: Automates the build and deployment of MkDocs to GitHub Pages using `mike` for versioning.  
*   **Link Check**: Periodically checks for broken links in the documentation using `lycheeverse/lychee-action`.  

## 🛠 Skills

Nutze bei Bedarf die Skills aus [dgaida/auto-version-action](https://github.com/dgaida/auto-version-action/tree/main/skills):
*   **`SKILL_coding.md`**: Nutze diesen Skill für tiefe Code-Reviews, Audits, Analysen oder Refactoring-Pläne. Er hilft dabei, die Wartbarkeit, Klarheit und Korrektheit des Codes zu verbessern.
*   **`SKILL_docs.md`**: Nutze diesen Skill für alle Aufgaben rund um die Dokumentation (MkDocs, API-Extraktion, Docstrings, Versionierung). Er unterstützt bei der Erstellung eines vollständigen, zweisprachigen Dokumentations-Ökosystems.
