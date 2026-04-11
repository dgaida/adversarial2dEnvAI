# Installation

This section explains how to install the `custom_grid_env` library on your system.

## Prerequisites

Make sure you have Python 3.8 or newer installed. It is recommended to use a virtual environment.

## Installation from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/dgaida/adversarial2dEnvAI.git
   cd adversarial2dEnvAI
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

This will install all necessary dependencies such as `gymnasium`, `numpy`, `pygame`, `tensorflow`, `matplotlib`, and `scikit-learn`.

## Verification

You can verify the installation by trying to import the package in Python:

```python
import custom_grid_env
print(custom_grid_env.__version__)
```
