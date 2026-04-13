# Adversarial Search: Minimax & Expectimax

This tutorial explains the implementation and use of Minimax and Expectimax algorithms in the CustomGrid environment.

## Minimax Algorithm

The Minimax algorithm is used in zero-sum games to find the maximum gain (or minimum loss) for a player, assuming that the opponent also plays optimally.

### Alpha-Beta Pruning
To make the search more efficient, our implementation uses Alpha-Beta pruning. This allows ignoring branches in the search tree that cannot affect the final result.

### Usage in CustomGrid
The `MinimaxAgent` evaluates states based on a heuristic that minimizes the distance to the goal and maximizes the distance to the ghost.

```python
from custom_grid_env.agents.adversarial_agents import MinimaxAgent

# Initialize the agent with a search depth of 4
agent = MinimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=4)
```

## Expectimax Algorithm

In stochastic environments (like CustomGrid with slip probability), Minimax is often too pessimistic. Expectimax replaces the min nodes (or opponent nodes) with expectation nodes.

### Probabilities
Expectimax calculates the weighted average of the values of all possible successor states based on the slip probability.

```python
from custom_grid_env.agents.adversarial_agents import ExpectimaxAgent

# Initialize the Expectimax agent
agent = ExpectimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=3)
```

## Comparison

| Algorithm | Best Application | Considers Stochasticity |
|-----------|------------------|-------------------------|
| **Minimax** | Deterministic, optimal opponent | No (assumes worst-case) |
| **Expectimax** | Stochastic, average opponent | Yes |

## Heuristic Function

Both agents use an internal heuristic function:
- **Goal reached**: +10,000
- **Caught by ghost**: -10,000
- **Distance to goal**: Penalizes large distances.
- **Distance to ghost**: Rewards safety buffers.
