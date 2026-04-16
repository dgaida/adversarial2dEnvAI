# Adversarial Search: Minimax & Expectimax

Dieses Tutorial erklärt die Implementierung und Verwendung der Minimax- und Expectimax-Algorithmen in der CustomGrid-Umgebung.

## Minimax-Algorithmus

Der Minimax-Algorithmus wird in Nullsummenspielen verwendet, um den maximalen Gewinn (oder den minimalen Verlust) für einen Spieler zu finden, unter der Annahme, dass der Gegner ebenfalls optimal spielt.

### Alpha-Beta-Pruning
Um die Suche effizienter zu gestalten, verwendet unsere Implementierung Alpha-Beta-Pruning. Dies ermöglicht es, Zweige im Suchbaum zu ignorieren, die das Endergebnis nicht beeinflussen können.

### Verwendung in CustomGrid
Der `MinimaxAgent` bewertet Zustände basierend auf einer Heuristik, die die Distanz zum Ziel minimiert und die Distanz zum Geist maximiert.

```python
from custom_grid_env.agents.adversarial_agents import MinimaxAgent

# Initialisierung des Agenten mit einer Suchtiefe von 4
agent = MinimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=4)
```

## Expectimax-Algorithmus

In stochastischen Umgebungen (wie CustomGrid mit Rutschwahrscheinlichkeit) ist Minimax oft zu pessimistisch. Expectimax ersetzt die Min-Knoten (oder Gegner-Knoten) durch Erwartungswert-Knoten.

### Wahrscheinlichkeiten
Expectimax berechnet den gewichteten Durchschnitt der Werte aller möglichen Nachfolgezustände basierend auf der Rutschwahrscheinlichkeit.

```python
from custom_grid_env.agents.adversarial_agents import ExpectimaxAgent

# Initialisierung des Expectimax-Agenten
agent = ExpectimaxAgent(interface.get_action_space(), env=interface.env, depth_limit=3)
```

## Umgebungsannahmen

### Nicht-zyklische Umgebung
Es wird davon ausgegangen, dass die Umgebung **nicht-zyklisch** ist. Das bedeutet, es gibt keine "Wrap-around"-Bewegungen. Sowohl Minimax- als auch Expectimax-Agenten berücksichtigen dies explizit, indem sie die `_move_entity`-Logik der Umgebung während der Zustandssimulation nutzen.

### Konsequenzen für die Strategie
Die nicht-zyklische Natur des Gitters hat mehrere strategische Auswirkungen auf die adversarielle Suche:

- **Einkesseln (Cornering)**: Da das Gitter feste Grenzen hat, kann der Geist (wenn er Minimax verwendet) den Agenten effektiv in einer Ecke oder an einer Wand festsetzen. Der Agent wiederum muss vermeiden, in eine Position zu geraten, in der seine Bewegungsoptionen eingeschränkt sind.
- **Grenzkollisionen**: Bei der Simulation zukünftiger Züge erkennen die Agenten, dass Aktionen, die "aus dem Spielfeld" führen würden, dazu führen, dass die Einheit in ihrem aktuellen Feld bleibt. Dies ist entscheidend für eine genaue Wertschätzung in beiden Algorithmen.
- **Vereinfachte Distanz**: Heuristiken wie die Manhattan-Distanz oder die BFS-basierte kürzeste Pfadlänge sind unkomplizierter, da sie keine modulare Arithmetik oder Wrap-around-Pfade berücksichtigen müssen.

## Vergleich

| Algorithmus | Beste Anwendung | Berücksichtigt Stochastik |
|-----------|------------------|-------------------------|
| **Minimax** | Deterministisch, optimaler Gegner | Nein (geht vom Worst-Case aus) |
| **Expectimax** | Stochastisch, durchschnittlicher Gegner | Ja |

## Heuristik-Funktion

Beide Agenten verwenden eine interne Heuristik-Funktion:  
- **Ziel erreicht**: +10.000  
- **Vom Geist gefangen**: -10.000  
- **Distanz zum Ziel**: Bestraft große Distanzen.  
- **Distanz zum Geist**: Belohnt Sicherheitsabstände.
