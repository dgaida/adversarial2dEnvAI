# CustomGrid Environment

Eine Gymnasium-basierte Gitterumgebung, in der ein Agent durch ein 4x5-Gitter navigiert, um Zielzellen zu erreichen, während er einem jagenden Geist ausweicht.

![Umgebungs-Vorschau](../Umgebung.png)

## Überblick

CustomGrid ist eine rundenbasierte Umgebung, in der:
- Ein **Agent** (Roboter mit GPS) versucht, eine der Zielzellen zu erreichen.
- Ein **Geist** den Agenten in jeder Runde jagt.
- **Wände** die Bewegung zwischen bestimmten Zellen blockieren.
- **Rutschwahrscheinlichkeit** für Stochastik sorgt – der Agent kann senkrecht zur beabsichtigten Richtung rutschen.
- **Farbige Zellen** visuelle Informationen liefern (rote und grüne Muster).

## Schnellstart

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent

# Erstelle das Interface
interface = AgentInterface(render=True, slip_probability=0.2)

# Zurücksetzen und erste Beobachtung erhalten
obs = interface.reset()

# Erstelle deinen Agenten
agent = RandomPlayerAgent(interface.get_action_space())

# Eine Episode ausführen
while not interface.is_terminated():
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)

# Ergebnisse abrufen
stats = interface.get_episode_stats()
print(f"Gesamtbelohnung: {stats['total_reward']}")

interface.close()
```

## Voraussetzungen

- Python 3.8+
- gymnasium
- numpy
- pygame
