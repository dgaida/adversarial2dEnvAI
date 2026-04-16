# Adversarial Search: Minimax & Expectimax

Dieses Tutorial erklärt die Implementierung und Nutzung von Minimax- und Expectimax-Algorithmen in der CustomGrid-Umgebung.

## Minimax-Algorithmus

Der Minimax-Algorithmus wird in Nullsummenspielen verwendet, um den maximalen Gewinn (oder minimalen Verlust) für einen Spieler zu finden, unter der Annahme, dass der Gegner ebenfalls optimal spielt.

### Alpha-Beta-Pruning
Um die Suche effizienter zu gestalten, verwendet unsere Implementierung Alpha-Beta-Pruning. Dies ermöglicht es, Äste im Suchbaum zu ignorieren, die das Endergebnis nicht beeinflussen können.

### Verwendung in CustomGrid
Der `MinimaxAgent` bewertet Zustände basierend auf einer Heuristik, die die Distanz zum Ziel minimiert und die Distanz zum Geist maximiert.

```python
from custom_grid_env.agents.adversarial_agents import MinimaxAgent

# Initialisiere den Agenten mit einer Suchtiefe von 4
agent = MinimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=4)
```

## Expectimax-Algorithmus

In stochastischen Umgebungen (wie CustomGrid mit Rutschwahrscheinlichkeit) ist Minimax oft zu pessimistisch. Expectimax ersetzt die Min-Knoten (oder Gegner-Knoten) durch Erwartungswert-Knoten.

### Wahrscheinlichkeiten
Expectimax berechnet den gewichteten Durchschnitt der Werte aller möglichen Folgezustände basierend auf der Rutschwahrscheinlichkeit.

**Wichtige Annahme**: Der Expectimax-Agent geht davon aus, dass der Geist **zufällig** (RandomGhostAgent) agiert, anstatt optimal zu spielen. Dies macht ihn in stochastischen Umgebungen oft effektiver als den pessimistischen Minimax-Agenten, solange der Geist nicht perfekt spielt.

```python
from custom_grid_env.agents.adversarial_agents import ExpectimaxAgent

# Initialisiere den Expectimax-Agenten
agent = ExpectimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=3)
```

## Vergleich

| Algorithmus | Beste Anwendung | Berücksichtigt Stochastik |
|-------------|-----------------|---------------------------|
| **Minimax** | Deterministisch, optimaler Gegner | Nein (geht vom Worst-Case aus) |
| **Expectimax** | Stochastisch, durchschnittlicher Gegner | Ja |

## Heuristik-Funktion

Beide Agenten verwenden eine interne Heuristik-Funktion:  
- **Ziel erreicht**: +10.000  
- **Vom Geist gefangen**: -10.000  
- **Distanz zum Ziel**: Bestraft große Distanzen.  
- **Distanz zum Geist**: Belohnt Sicherheitspuffer.

### Interaktives Ausprobieren

Nutzen Sie das `Colab_GUI_Demo` Notebook, um die Agenten live zu vergleichen:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb)

### Übungen für Studierende:
1. **Vergleich**: Stellen Sie die Suchtiefe auf 3 und vergleichen Sie das Überlebensvermögen von Minimax und Expectimax gegen einen `ChaseGhostAgent`.
2. **Geist-Verhalten**: Ändern Sie das Verhalten des Geistes auf `Minimax`. Welcher Agent (Minimax vs. Expectimax) schneidet nun besser ab?
3. **Sichtbarkeit**: Aktivieren Sie "Estimated State" in der GUI. Wie beeinflusst die Unsicherheit der Lokalisierung (Partikelfilter) die Qualität der Spielzüge der adversarialen Agenten?
