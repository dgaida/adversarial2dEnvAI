# Erste Schritte

Willkommen bei `custom_grid_env`! Diese Seite hilft dir dabei, dich schnell in der Umgebung zurechtzufinden.

## Die Konzepte

Die Umgebung simuliert ein Gitter, in dem ein Agent navigieren muss. Dabei spielen folgende Komponenten eine zentrale Rolle:

1.  **AgentInterface**: Die primäre Schnittstelle für deine KI-Agenten. Sie kapselt die Umgebung, den Geist und den Partikelfilter.
2.  **Partikelfilter**: Ein Mechanismus zur Schätzung der Agentenposition, falls diese nicht exakt bekannt ist (Lokalisierung).
3.  **CNN-Klassifizierung**: Ein neuronales Netz, das Bilder der Gitterzellen verarbeitet, um Objekte wie Hunde oder Blumen zu erkennen.

## Erste Experimente

Der beste Weg, um zu starten, sind unsere interaktiven Jupyter Notebooks:

- **Umgebungs-Demo**: Lerne die Grundlagen der Steuerung kennen.
- **Interaktive GUI**: Experimentiere mit verschiedenen Sensoren und dem Partikelfilter direkt in Colab.
- **CNN-Training**: Erfahre, wie man das neuronale Netz trainiert, das der Agent zur Objekterkennung nutzt.

## Ein einfaches Beispiel

Hier ist ein minimales Skript, um einen Agenten mit zufälligen Bewegungen zu starten:

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent

# Interface initialisieren
interface = AgentInterface(render=True)
obs = interface.reset()

# Agent erstellen
agent = RandomPlayerAgent(interface.get_action_space())

# Episode ausführen
for _ in range(100):
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)
    if done:
        break

interface.close()
```

## Weiterführende Tutorials

Schau dir unsere detaillierten Tutorials an:
- [CNN Training Tutorial](tutorial/cnn.md)
- [Partikelfilter Tutorial](tutorial/particle_filter.md)
