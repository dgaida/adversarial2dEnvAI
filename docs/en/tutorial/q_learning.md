# Q-Learning Tutorial

Q-Learning is a model-free Reinforcement Learning algorithm. The agent learns the "quality" (Q-value) of actions in specific states directly from interacting with the environment.

## Concept

Unlike Value Iteration, Q-Learning does not require knowledge of the environment's rules (transition probabilities). The agent tries actions and updates its estimation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

- **Exploration vs. Exploitation**: With the epsilon-greedy strategy, the agent sometimes chooses random actions ($\epsilon$) to discover new things, and mostly the best-known action (-\epsilon$).  

## Interactive Notebook

Get hands-on experience with Q-Learning and train your own agent:

[![Open In Colab](../../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Q_Learning.ipynb)

## Visualization in the GUI

In the `ColabGUI`, you can select the Q-Learning agent. The GUI visualizes the learned Q-values directly in the grid:  
- The highest Q-value of a cell is displayed.  
- The position of the value in the cell (top, bottom, left, right) indicates the currently preferred direction.  
