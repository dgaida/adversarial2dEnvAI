# Task Planning & LLM-Integration

Dieses Tutorial beschreibt, wie die CustomGrid-Umgebung natürliche Sprache nutzt, um komplexe Aufgaben zu verstehen, Ziele zu identifizieren und diese effizient anzusteuern.

## Übersicht des Workflows

Der Prozess der Aufgabenplanung besteht aus drei Hauptschritten:
1. **Zielidentifikation**  : Ein LLM (Large Language Model) extrahiert Koordinaten aus einer natürlichsprachlichen Beschreibung.
2. **Reihenfolge-Optimierung**  : Ein TSP-Solver (Traveling Salesperson Problem) bestimmt die kürzeste Route, um alle Ziele zu besuchen.
3. **Pfadplanung**  : Value Iteration berechnet die optimalen Bewegungen, um jedes Ziel nacheinander zu erreichen.

## 1. Zielidentifikation mit LLMs

Die Klasse `TaskPlanner` nutzt ein LLM (standardmäßig über Groq), um Benutzeranweisungen wie *"Besuche erst den Hund und dann die Blumen"* zu interpretieren.

### Grid-Beschreibung
Damit das LLM weiß, wo sich Objekte befinden, generiert die Umgebung eine Textbeschreibung des aktuellen Gitters (`get_grid_description()`). Dabei werden Items wie folgt übersetzt:
- **dog**: "ein Bild eines Hundes"
- **flower**: "eine Blume"
- **two_flowers**: "zwei Blumen"

### Prompting & Extraktion
Das LLM erhält einen System-Prompt, der es anweist, **ausschließlich** ein JSON-Array von Koordinaten zurückzugeben (z. B. `[[0, 1], [2, 3]]`).

Da moderne Modelle oft "Gedankengänge" (Reasoning) in `<think>`-Tags ausgeben, enthält der `TaskPlanner` eine robuste Reinigungslogik:
- Entfernen von `<think>`-Blöcken mittels Regex.
- Extraktion des JSON-Arrays aus dem verbleibenden Text.
- Validierung des Formats, um Parsing-Fehler zu vermeiden.

## 2. Optimierung der Route (TSP)

Sobald die Zielkoordinaten feststehen, muss die effizienteste Reihenfolge gefunden werden.

### Distanzmatrix
Der `TaskPlanner` berechnet zunächst eine Distanzmatrix zwischen dem Startpunkt und allen Zielen. Da die Umgebung Wände enthält, wird hierfür eine Breitensuche (BFS) genutzt, um die echte "Shortest Path Distance" zu ermitteln.

### TSP-Solver
Der Solver probiert alle Permutationen der Zielreihenfolge aus und wählt diejenige mit der geringsten Gesamtdistanz (inklusive Rückkehr zum Startpunkt). Bei der geringen Anzahl an Zielen in diesem Szenario ist dieser brute-force Ansatz extrem schnell und garantiert die globale Optimalität.

## 3. Pfadplanung mittels Value Iteration

Um ein spezifisches Ziel im Gitter zu erreichen, wird **Value Iteration** eingesetzt.

- **Zustandswerte**: Jede Zelle erhält einen Wert basierend auf der Distanz zum Ziel. Das Ziel selbst hat den höchsten Wert (+100).
- **Iterative Updates**: Die Werte werden so lange aktualisiert, bis sie konvergieren (`theta < 0.0001`).
- **Aktionswahl**: Der Agent wählt in jedem Schritt die Aktion, die in das benachbarte Feld mit dem höchsten Wert führt.

Dieser Ansatz ist robust gegenüber den Wänden der Umgebung und garantiert den kürzesten Weg zum Ziel.

## Integration in die Colab-GUI

In der `ColabGUI` ist dieser Prozess in zwei Schritte unterteilt:
1. **Plan**  : Das LLM wird aufgerufen, die Ziele werden visualisiert und die optimale Tour wird als Checkliste angezeigt.
2. **Execute**  : Der Agent arbeitet die Liste der Ziele nacheinander ab, wobei der Fortschritt in Echtzeit in der GUI markiert wird.

### Übung für Studierende
1. **Mehrdeutigkeit**  : Geben Sie eine unklare Anweisung ein (z. B. "Gehe zu etwas Schönem"). Wie reagiert das LLM basierend auf der Grid-Beschreibung?
2. **Komplexität**  : Erstellen Sie eine Aufgabe mit 4-5 Zielen. Beobachten Sie, wie der TSP-Solver die Reihenfolge im Vergleich zur Nennung im Text ändert.
3. **Hindernisse**  : Platzieren Sie Wände so, dass ein direkter Weg blockiert ist. Verifizieren Sie, dass die Value Iteration dennoch den optimalen Umweg findet.
