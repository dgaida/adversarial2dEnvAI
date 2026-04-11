# CustomGrid Environment

Eine Gymnasium-basierte Gitterumgebung, in der ein Agent durch ein 4x5-Gitter navigiert, um Zielzellen zu erreichen, während er einem jagenden Geist ausweicht.

## Interaktive Notebooks

Sie können das CNN-Training und die Umgebung direkt in Ihrem Browser mit Google Colab ausprobieren:

- **CNN-Trainings-Tutorial**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb)
- **Umgebungs-Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb)
- **Interaktive GUI**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb)

![Umgebungs-Vorschau](../Umgebung.png)

## Überblick

CustomGrid ist eine rundenbasierte Umgebung, in der:
- Ein **Agent** (Roboter mit GPS) versucht, eine der Zielzellen zu erreichen.
- Ein **Geist** den Agenten in jeder Runde jagt.
- **Wände** die Bewegung zwischen bestimmten Zellen blockieren.
- **Rutschwahrscheinlichkeit** für Stochastik sorgt – der Agent kann senkrecht zur beabsichtigten Richtung rutschen.
- **Farbige Zellen** visuelle Informationen liefern (rote und grüne Muster).

## Ziel der Umgebung

Das Hauptziel der Umgebung ist die Entwicklung eines mobilen Agenten, der verschiedene Orte (die nur visuell oder akustisch detektiert werden können) in minimaler Zeit besuchen und zum Ausgangsort zurückkehren soll, ohne dabei mit anderen Verkehrsteilnehmern zu kollidieren (ähnlich dem Problem des Handlungsreisenden).

### Gegebene Ressourcen
- **Agent**: Ausgestattet mit Sensoren, einem Fahrwerk und einem kleinen Rechner.
- **Sensoren**: Kamera, Mikrofon, Farbsensor.
- **Karte**: Information darüber, in welchen Feldern welche optischen und akustischen Reize wahrgenommen werden können.

### Beispielaufgabe
Zu Beginn erhält der Agent eine Anweisung wie:
„Besuche in optimaler Reihenfolge die folgenden drei Felder und kehre zum Ausgangsort zurück:
- das Feld in dem man Klaviermusik hört,
- das Feld wo man das Bild des Hundes sieht und Rockmusik hört,
- das Feld mit dem Schriftzug ‚Ziel‘.“

**Wichtig**: Der Agent kennt zum Zeitpunkt $t=0$ seine Position nicht und muss diese mithilfe seiner Sensoren schätzen (Lokalisierung).

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
