# Lokalisierung

Die Umgebung bietet einen **Partikelfilter**, um die Position des Agenten zu schätzen, wenn dessen genauer Standort unbekannt ist oder wenn mit verrauschten Sensoren navigiert wird.

## Partikelfilter

Ein Partikelfilter (PF) ist ein Monte-Carlo-Algorithmus zur Schätzung des Zustands eines Systems. In dieser Umgebung wird er zur **Selbstlokalisierung** des Agenten auf dem Gitter verwendet.

### Funktionsweise

1. **Initialisierung**: Partikel werden zufällig über das gesamte Gitter verteilt.
2. **Prädiktion (Prediction)**: Wenn sich der Agent bewegt, wird jedes Partikel entsprechend der beabsichtigten Aktion bewegt, wobei die Bewegungsunsicherheit (Rutschen) berücksichtigt wird.
3. **Update**: Das Gewicht jedes Partikels wird basierend darauf aktualisiert, wie gut die hypothetischen Beobachtungen an der Position dieses Partikels mit den tatsächlichen Sensormessungen übereinstimmen.
4. **Resampling**: Partikel mit geringen Gewichten werden durch Kopien von Partikeln mit hohen Gewichten ersetzt.

## Sensoren

Die Lokalisierung stützt sich auf zwei Hauptsensortypen:

### Farbsensor
Der Farbsensor misst die Farbe des Bodens direkt unter dem Agenten.
- **Genauigkeit**: 80% (korrekte Farbe).
- **Rauschen**: 20% (gleichmäßig auf die anderen beiden Farben verteilt).
- **Farben**: Weiß (0), Rot (1), Grün (2).

### CNN-Klassifikator
Der CNN-Klassifikator wird bei jedem Schritt ausgeführt, unabhängig davon, ob ein Objekt vorhanden ist. Wenn kein Objekt vorhanden ist, wird die Klasse "Hintergrund" vorhergesagt.
- **Klassen**: Hund, Blume, Hintergrund.
- **Verwendung**: Die vom CNN zugewiesene Wahrscheinlichkeit für die Klasse, die sich tatsächlich in der Zelle eines Partikels befindet, wird als Likelihood für dieses Partikel verwendet.

## Sensor-Fusion

Bei Verwendung beider Sensoren (`sensor_mode='both'`) kombiniert der Filter die Messungen unter der Annahme, dass sie bei gegebenem Zustand bedingt unabhängig sind. Die Verbundwahrscheinlichkeit (Joint Likelihood) ist das Produkt der einzelnen Likelihoods:

$$p(z_{\text{color}}, z_{\text{cnn}} | s) = p(z_{\text{color}} | s) \cdot p(z_{\text{cnn}} | s)$$

## Annahmen

Der Partikelfilter trifft folgende Annahmen über die Umgebung:

### Kartenannahme (Map Assumption)
Der Filter verfügt über **perfektes Wissen über die Karte**. Er kennt:
- Die Dimensionen des Gitters.
- Die genaue Position aller Wände.
- Die Bodenfarbe jeder Zelle.
- Die Standorte aller Objekte (Hunde und Blumen).

### Bewegungsunsicherheit (Movement Uncertainty)
Der Filter geht von einem **stochastischen Bewegungsmodell** aus, das der Rutschwahrscheinlichkeit der Umgebung entspricht:
- **Beabsichtigte Bewegung**: Der Agent bewegt sich mit einer Wahrscheinlichkeit von $1 - P_{\text{slip}}$ in die beabsichtigte Richtung.
- **Rutschen (Slipping)**: Der Agent bewegt sich mit einer Wahrscheinlichkeit von $P_{\text{slip}}$ in eine senkrechte Richtung (gleichmäßig auf die beiden senkrechten Richtungen verteilt).
- **Wände**: Wenn eine Bewegung gegen eine Wand oder aus dem Gitter heraus führen würde, bleibt der Agent (und damit auch die Partikel) in der aktuellen Zelle.

### Messunsicherheit (Measurement Uncertainty)
Der Filter verwendet folgende **Likelihood-Modelle**:
- **Farbsensor**: $p(z_{\text{color}} | s) = 0,8$, wenn die gemessene Farbe $z$ mit der Kartenfarbe am Zustand $s$ übereinstimmt, andernfalls $0,1$.
- **CNN**: $p(z_{\text{cnn}} | s) = \text{CNN\_prob}(\text{Klasse an Stelle } s)$. Der Filter nimmt an, dass die vom CNN ausgegebene Wahrscheinlichkeit für die wahre Klasse an einem Ort die Likelihood dieses Ortes ist.
