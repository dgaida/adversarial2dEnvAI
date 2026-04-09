# Konfiguration

Die `custom_grid_env` kann über verschiedene Parameter angepasst werden, hauptsächlich über das `AgentInterface`.

## AgentInterface Parameter

Beim Initialisieren des `AgentInterface` können folgende Parameter übergeben werden:

| Parameter | Typ | Standard | Beschreibung |
|-----------|------|---------|-------------|
| `render` | bool | `True` | Aktiviert die grafische Anzeige via Pygame. |
| `step_delay` | int | `100` | Verzögerung in Millisekunden zwischen den Schritten bei der Anzeige. |
| `slip_probability` | float | `0.2` | Wahrscheinlichkeit (0.0 bis 1.0), dass der Agent rutscht. |
| `ghost_agent_class` | Type | `ChaseGhostAgent` | Klasse, die das Verhalten des Geistes steuert. |
