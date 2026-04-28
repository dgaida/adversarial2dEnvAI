# Value Iteration Tutorial

Value Iteration is a classic dynamic programming algorithm used to find the optimal value function ^*$. In this project, we use it for precise path planning.

## Concept

In Value Iteration, we iteratively calculate the value of each state (cell) based on the rewards of neighboring states and their own values.

The Bellman equation is at its core:
$$V(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]$$

In our grid, this means:  
- **Goal**: High positive value (100).  
- **Normal Step**: Small negative value (-1).  
- **Obstacles/Walls**: Prevent movement (state value remains low).  

## Interactive Notebook

You can follow and train the algorithm step-by-step in our interactive Jupyter notebook:

[![Open In Colab](../../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Value_Iteration.ipynb)

## Implementation in the Project

In the code, you can find the implementation in:  
- `src/custom_grid_env/planner.py`: Contains the logic for `value_iteration()`.  
- `src/custom_grid_env/agents/value_iteration_agent.py`: The agent that uses these values to select the best action.  
