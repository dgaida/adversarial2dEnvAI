# Rewards

## Reward Values

| Event | Reward | Terminal |
|-------|--------|----------|
| Each step (default) | -1 | No |
| Caught by ghost | -50 | **Yes** |
| Reached goal | +100 | **Yes** |

## Strategy

To maximize the reward, the agent should:
1. Find the shortest path to the goal.
2. Avoid the ghost at all costs.
