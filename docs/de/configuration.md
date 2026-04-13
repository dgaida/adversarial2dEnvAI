# Konfiguration

Die `custom_grid_env` kann über verschiedene Parameter angepasst werden, hauptsächlich über das `AgentInterface`.

## AgentInterface Parameter

Beim Initialisieren des `AgentInterface` können folgende Parameter übergeben werden:

| Parameter | Typ | Standard | Beschreibung |
|-----------|------|---------|-------------|
| `render` | bool | `True` | Aktiviert die grafische Anzeige via Pygame. |
| `render_mode` | str | `None` | Der Render-Modus ("human" oder "rgb_array"). Standardmäßig "rgb_array", wenn `render=True`. |
| `step_delay` | int | `100` | Verzögerung in Millisekunden zwischen den Schritten bei der Anzeige. |
| `slip_probability` | float | `0.2` | Wahrscheinlichkeit (0.0 bis 1.0), dass der Agent rutscht. |
| `slip_type` | str | `"longitudinal"` | Art des Rutschens ("longitudinal" oder "perpendicular"). |
| `ghost_agent_class` | Type | `ChaseGhostAgent` | Klasse, die das Verhalten des Geistes steuert. |
| `use_particle_filter` | bool | `True` | Ob der Partikelfilter zur Zustandsschätzung verwendet werden soll. |
| `pf_num_particles` | int | `200` | Anzahl der Partikel im Filter. |
| `pf_sensor_mode` | str | `"both"` | Sensormodus für den PF ("color", "cnn" oder "both"). |
| `show_particles` | bool | `True` | Ob Partikel im Renderer angezeigt werden sollen. |

## Beispiel Konfiguration

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
