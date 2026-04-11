# Gameplay Mechanics

## Turn System

The game uses an **alternating turn system**:

1. **Agent's Turn**: The agent chooses and executes an action.
2. **Ghost's Turn**: The ghost automatically moves toward the agent.

Each call to `interface.step(action)` processes both turns and returns the resulting state.

## Actions

Both the agent and ghost use the same action space:

| Action | Direction | Effect             |
|--------|-----------|--------------------|
| 0      | Left      | Move to column - 1 |
| 1      | Down      | Move to row + 1    |
| 2      | Right     | Move to column + 1 |
| 3      | Up        | Move to row - 1    |

## Slip Probability

The environment includes stochastic slip mechanics for the agent. With a configured probability $P_{\text{slip}}$, the agent moves differently than intended:

- **Perpendicular Slipping**: The agent moves in a perpendicular direction (split equally between the two perpendicular directions).
- **Longitudinal Slipping**: The agent moves in the same direction but either stays in place or moves twice.
