# Spielmechanik

## Turn System

Das Spiel verwendet ein **abwechselndes Rundensystem**:

1. **Agentenzug**: Der Agent wählt eine Aktion aus und führt sie aus.
2. **Geisterzug**: Der Geist bewegt sich automatisch gemäß seiner konfigurierten Strategie.

Jeder Aufruf von `interface.step(action)` verarbeitet beide Züge und gibt den resultierenden Zustand zurück.

## Agenten-Verhalten

Sie können verschiedene Agenten-Typen verwenden, um das Gitter zu navigieren:

- **RandomPlayerAgent**: Wählt in jedem Zug zufällig eine der vier möglichen Aktionen aus.
- **MinimaxAgent**: Nutzt den Minimax-Algorithmus mit Alpha-Beta-Pruning, um optimale Züge unter Berücksichtigung des Geistes zu finden.
- **ExpectimaxAgent**: Ähnlich wie Minimax, berücksichtigt aber die Stochastik der Umgebung (Rutschen).

## Geister-Verhalten

Das Verhalten des Gespensts kann über den `AgentInterface` oder die GUI angepasst werden. Folgende Implementierungen stehen zur Verfügung:

- **ChaseGhostAgent (Standard)**: Der Geist berechnet den kürzesten Pfad zum Agenten (unter Berücksichtigung von Wänden) und bewegt sich einen Schritt in diese Richtung.
- **RandomGhostAgent**: Der Geist wählt in jedem Zug zufällig eine der vier möglichen Aktionen aus.
- **MinimaxAgent (als Geist)**: Der Geist kann ebenfalls Minimax verwenden, um den Agenten aktiv in die Enge zu treiben.

## Aktionen

Sowohl der Agent als auch der Geist nutzen denselben Aktionsraum:

| Aktion | Richtung | Effekt             |
|--------|-----------|--------------------|
| 0      | Links      | Spalte - 1 |
| 1      | Runter     | Zeile + 1    |
| 2      | Rechts     | Spalte + 1 |
| 3      | Hoch        | Zeile - 1    |

## Rutschwahrscheinlichkeit (Slip)

Die Umgebung enthält eine stochastische Rutschmechanik für den Agenten. Bei einer konfigurierten Wahrscheinlichkeit $P_{\text{slip}}$ bewegt sich der Agent anders als beabsichtigt:

- **Senkrechtes Rutschen (Perpendicular Slipping)**: Der Agent bewegt sich in eine senkrechte Richtung (gleichmäßig auf die beiden senkrechten Richtungen verteilt).
- **Längsrutschen (Longitudinal Slipping)**: Der Agent bewegt sich in die gleiche Richtung, bleibt aber entweder stehen oder bewegt sich doppelt so weit.
