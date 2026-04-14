# Configuration

The `custom_grid_env` can be customized via various parameters, primarily through the `AgentInterface`.

## AgentInterface Parameters

When initializing the `AgentInterface`, the following parameters can be passed:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `render` | bool | `True` | Enables graphical display via Pygame. |
| `render_mode` | str | `None` | The mode to render with ("human" or "rgb_array"). Defaults to "rgb_array" if `render=True`. |
| `step_delay` | int | `100` | Delay in milliseconds between steps when rendering. |
| `slip_probability` | float | `0.2` | Probability (0.0 to 1.0) of the agent slipping. |
| `slip_type` | str | `"longitudinal"` | Type of slipping ("longitudinal" or "perpendicular"). |
| `ghost_agent_class` | Type | `ChaseGhostAgent` | Class that controls the ghost's behavior. |
| `use_particle_filter` | bool | `True` | Whether to use the particle filter for state estimation. |
| `pf_num_particles` | int | `200` | Number of particles in the filter. |
| `pf_sensor_mode` | str | `"both"` | Sensor mode for the PF ("color", "cnn", or "both"). |
| `show_particles` | bool | `True` | Whether to show particles in the renderer. |

## Example Configuration

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_ghost_agent import RandomGhostAgent

interface = AgentInterface(
    render=True,
    slip_probability=0.1,
    slip_type="perpendicular",
    ghost_agent_class=RandomGhostAgent,
    pf_num_particles=500
)
```
