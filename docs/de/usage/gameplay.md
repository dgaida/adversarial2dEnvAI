# Spielmechanik

## Turn System

Das Spiel verwendet ein **abwechselndes Rundensystem**:

1. **Agentenzug**: Der Agent wählt eine Aktion aus und führt sie aus.
2. **Geisterzug**: Der Geist bewegt sich automatisch in Richtung des Agenten.

Jeder Aufruf von `interface.step(action)` verarbeitet beide Züge und gibt den resultierenden Zustand zurück.

## Aktionen

Sowohl der Agent als auch der Geist nutzen denselben Aktionsraum:

| Aktion | Richtung | Effekt             |
|--------|-----------|--------------------|
| 0      | Links      | Spalte - 1 |
| 1      | Runter     | Zeile + 1    |
| 2      | Rechts     | Spalte + 1 |
| 3      | Hoch        | Zeile - 1    |

## Rutschwahrscheinlichkeit (Slip)

Die Umgebung enthält eine stochastische Rutschmechanik für den Agenten. Bei einer konfigurierten Wahrscheinlichkeit bewegt sich der Agent stattdessen in eine **senkrechte Richtung**.
