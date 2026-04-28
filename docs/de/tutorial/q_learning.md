# Q-Learning Tutorial

Q-Learning ist ein modell-freier Reinforcement Learning Algorithmus. Der Agent lernt hierbei die "Qualität" (Q-Wert) von Aktionen in bestimmten Zuständen direkt aus der Interaktion mit der Umgebung.

## Konzept

Im Gegensatz zur Value Iteration benötigt Q-Learning kein Wissen über die Regeln der Umgebung (Transitionswahrscheinlichkeiten). Der Agent probiert Aktionen aus und aktualisiert seine Schätzung:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

- **Exploration vs. Exploitation**: Mit der Epsilon-Greedy-Strategie wählt der Agent manchmal zufällige Aktionen ($\epsilon$), um Neues zu entdecken, und meistens die beste bekannte Aktion (-\epsilon$).  

## Interaktives Notebook

Lerne Q-Learning praxisnah kennen und trainiere deinen eigenen Agenten:

[![Open In Colab](../../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Q_Learning.ipynb)

## Visualisierung in der GUI

In der `ColabGUI` kannst du den Q-Learning Agenten auswählen. Die GUI visualisiert die gelernten Q-Werte direkt im Grid:  
- Der höchste Q-Wert einer Zelle wird angezeigt.  
- Die Position des Wertes im Feld (oben, unten, links, rechts) zeigt die aktuell bevorzugte Richtung an.  
