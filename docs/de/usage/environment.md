# Gitter-Umgebung

Das CustomGrid ist ein 4x5 Gitter mit speziellen Feldern, Gegenständen und Hindernissen.

## Gitter-Layout

Das Gitter besteht aus 4 Zeilen (0-3) und 5 Spalten (0-4).

## Zellen-Farben

Einige Zellen haben Hintergrundfarben, die der Agent wahrnehmen kann:  
- **Rot**: Wird für bestimmte Muster verwendet.  
- **Grün**: Wird für bestimmte Muster verwendet.  
- **Weiß**: Standardfarbe.  

## Gegenstände

In der Welt sind verschiedene Gegenstände verteilt:  
- **Hund (dog)**: Wird von einem CNN klassifiziert, wenn der Agent das Feld betritt.  
- **Blume (flower)**: Wird von einem CNN klassifiziert, wenn der Agent das Feld betritt.  
- **Noten (notes)**: Werden in der Beobachtung (Observation) angezeigt.  

## Wände

Wände blockieren die Bewegung zwischen Zellen.  
- **Horizontale Wände**: Blockieren Auf/Ab Bewegungen.  
- **Vertikale Wände**: Blockieren Links/Rechts Bewegungen.  

## Aufgabenplanung (Task Planning)

Die Umgebung ist eng mit dem `TaskPlanner` verzahnt, um komplexe Missionen zu ermöglichen. Der `TaskPlanner` nutzt:  
- **Grid-Beschreibungen**: Automatisch generierte Texte über den Zustand der Welt.  
- **LLM-Integration**: Extraktion von Zielen aus natürlicher Sprache.  
- **Optimierung**: TSP-Solver für effiziente Routen und Value Iteration für die Pfadfindung.  

Weitere Details finden Sie im [Task Planning Tutorial](../tutorial/task_planning.md).
