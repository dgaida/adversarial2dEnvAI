# Beobachtungen (Observations)

## Struktur

Die Beobachtung ist ein verschachteltes Dictionary:

- `current_cell`: Info über die aktuelle Zelle (Farbe, Gegenstände, Ziel-Status, Text).
- `neighbors`: Erreichbarkeit und Farbe der 4 Nachbarzellen.
- `ghost_relative_pos`: Relative Position des Geistes zum Agenten `[row_diff, col_diff]`.

## Details

| Feld | Typ | Beschreibung |
|-------|------|-------------|
| `colour` | int | 0=weiß, 1=rot, 2=grün |
| `is_goal` | int | 1 wenn Zielzelle, sonst 0 |
| `accessible` | int | 1 wenn begehbar, 0 wenn durch Wand oder Rand blockiert |

## Info Dictionary

Zusätzlich zu den Beobachtungen gibt das Environment ein `info` Dictionary zurück, das folgende zusätzliche Informationen enthalten kann:

- `cnn_prediction`: Ein Tupel `(Klassenname, Wahrscheinlichkeit)`, falls sich der Agent auf einem Feld mit einem Hund oder einer Blume befindet und ein trainiertes Modell geladen wurde.
- `intended_action`: Die vom Agenten beabsichtigte Aktion.
- `actual_action`: Die tatsächlich ausgeführte Aktion (kann bei Rutschen abweichen).
