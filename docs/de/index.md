# CustomGrid Environment 🤖👻

![CustomGrid Environment](../Umgebung.png)

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/adversarial2dEnvAI/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![interrogate](../assets/interrogate.svg)](metrics.md)

Eine fortschrittliche Gymnasium-basierte Gitterumgebung für Reinforcement Learning und Robotik-Tutorials, die in der **KI-Vorlesung an der TH Köln** eingesetzt wird. CustomGrid bietet einen Agenten, der in einer stochastischen Umgebung mit unvollkommenen Sensoren, adversarialen Elementen und komplexer Zustandsschätzung navigiert.

## 🎯 Ziel der Umgebung

Das Hauptziel dieser Umgebung ist es, Studierenden zu vermitteln, wie man einen autonomen Agenten entwickelt, der in der Lage ist, komplexe, vom Benutzer definierte Aufgaben zu erfüllen. Diese Aufgaben können darin bestehen, bestimmte Zellen, die durch visuelle oder akustische Reize identifiziert werden, in einer optimalen Reihenfolge zu besuchen und zum Ausgangspunkt zurückzukehren, wobei oft von einem unbekannten Ort ausgegangen wird.

Eine zentrale Herausforderung ist die Integration mehrerer Module – wie **Speech-to-Text** für das Aufgabenverständnis, **Bild- und Akustiksensoren** zur Identifizierung und **Bayesianische Filter** zur Lokalisierung – um es dem Agenten zu ermöglichen, High-Level-Ziele zu lösen (z. B. „Besuche den Hund, dann das Ziel“).

## 🌟 Hauptmerkmale

*   **Rundenbasiertes Adversarial Gameplay**: Ein Agent konkurriert gegen einen Geist in einem 4x5 Gitter.  
*   **Adversarial Search**: Integrierte **Minimax** und **Expectimax** Agenten für strategische Planung.  
*   **Task Planning**: LLM-gestützte Zielidentifikation und TSP-Routenoptimierung.  
*   **Stochastische Bewegung**: Unterstützung für *senkrechtes* und *längsgerichtetes* Rutschen.  
*   **Unvollkommene Wahrnehmung**:  
    *   **Verrauschter Farbsensor**: Erkennung der Bodenfarbe mit 80% Genauigkeit.  
    *   **CNN-basierte Vision**: Echtzeit-Klassifizierung von Objekten (Hunde, Blumen, Hintergrund) mit einem vortrainierten CNN.  
*   **Zustandsschätzung**: Integrierter **Partikelfilter** für Bayes'sche Lokalisierung.  
*   **Interaktive Visualisierung**:  
    *   Umfangreicher Pygame-basierter Renderer.  
    *   Interaktive Google Colab GUI mit Echtzeit-2D-Wahrscheinlichkeitsverteilung.  
*   **Anpassbare Geister-KI**: Wechseln Sie zwischen kürzestem Pfad (Chase), Zufallsbewegung und Minimax.  

## 📓 Interaktive Notebooks

Erleben Sie die Umgebung direkt in Ihrem Browser:

| Notebook | Beschreibung | Link |
| :--- | :--- | :--- |
| **Interaktive GUI** | Vollständiges Dashboard mit Sensoren und PF-Visualisierung. | [![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb) |
| **Umgebungs-Demo** | Grundlegende programmatische Interaktion und API-Durchgang. | [![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Environment_Demo.ipynb) |
| **CNN-Training** | Tutorial zum Trainieren des Vision-Modells. | [![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb) |

## 🚀 Schnellstart

### Installation

```bash
pip install git+https://github.com/dgaida/adversarial2dEnvAI.git
```

### Grundlegende Nutzung

```python
from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.adversarial_agents import MinimaxAgent

# Interface mit Partikelfilter und Rendering initialisieren
interface = AgentInterface(render=True, slip_probability=0.2)
obs = interface.reset()

agent = MinimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=4)

while not interface.is_terminated():
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)

    # Zugriff auf geschätzte Position vom Partikelfilter
    est_pos = info['estimated_pos']['cell_pos']
    print(f"Geschätzte Position: {est_pos}")

interface.close()
```
