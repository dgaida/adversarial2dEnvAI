# Configuration

The `custom_grid_env` can be customized via various parameters, primarily through the `AgentInterface`.

## AgentInterface Parameters

When initializing the `AgentInterface`, the following parameters can be passed:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `render` | bool | `True` | Enables graphical display via Pygame. |
| `step_delay` | int | `100` | Delay in milliseconds between steps when rendering. |
| `slip_probability` | float | `0.2` | Probability (0.0 to 1.0) of the agent slipping. |
| `ghost_agent_class` | Type | `ChaseGhostAgent` | Class that controls the ghost's behavior. |
