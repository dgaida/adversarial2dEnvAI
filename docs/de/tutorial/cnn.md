# Tutorial: Training eines CNN zur Objekterkennung

Dieses Tutorial erklärt, wie man ein einfaches Convolutional Neural Network (CNN) trainiert, um Objekte in der CustomGrid-Umgebung zu klassifizieren. Dies ist ein idealer Einstieg für Studierende, um die Grundlagen von Computer Vision und deren Integration in Software-Systeme zu erlernen.

## Überblick

In unserer Umgebung kann der Agent auf "Hunde" (dogs) und "Blumen" (flowers) treffen. Während die Umgebung intern weiß, um welches Objekt es sich handelt, nutzen wir ein CNN, um die Objekte basierend auf ihrer visuellen Darstellung zu "sehen" und zu klassifizieren.

Der Prozess besteht aus drei Hauptschritten:
1. **Datengenerierung**: Erstellung eines Bilddatensatzes.
2. **Training**: Das neuronale Netz lehren, Muster zu erkennen.
3. **Integration**: Nutzung des trainierten Modells in der Umgebung.

### Interaktives Lernen

Sie können dieses Tutorial interaktiv mit Google Colab verfolgen:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb)

## 1. Datengenerierung

Bevor wir ein Modell trainieren können, benötigen wir Daten. Wir nutzen `src/custom_grid_env/cnn_tutorial/data_generation.py`, um prozedural Bilder mit einer Auflösung von 64x64 Pixeln zu erstellen.

### Funktionsweise:
- **Verschiedene Hintergründe**: Um das Modell robust zu machen, generieren wir die Objekte auf weißem, rot-schraffiertem und grün-schraffiertem Hintergrund.
- **Zufälligkeit**: Wir fügen kleine zufällige Verschiebungen der Position des Hundes oder der Blume hinzu.
- **Ausgabe**: Die Bilder werden in den Verzeichnissen `data/dog/` und `data/flower/` gespeichert.

Um die Daten zu generieren, führen Sie folgenden Befehl aus:
```bash
python src/custom_grid_env/cnn_tutorial/data_generation.py
```

## 2. Training des neuronalen Netzes

Die Trainingslogik befindet sich in `src/custom_grid_env/cnn_tutorial/train.py`. Wir verwenden **TensorFlow** und **Keras**, um unser Modell aufzubauen und zu trainieren.

### Die CNN-Architektur
Ein CNN ist speziell für die Verarbeitung von Pixeldaten konzipiert. Unser einfaches Modell nutzt folgende Schichten:
- **Conv2D**: Diese Schicht "schiebt" Filter über das Bild, um Merkmale wie Kanten oder Formen zu erkennen.
- **MaxPooling2D**: Dies reduziert die räumliche Größe der Darstellung, was das Modell schneller und robuster gegenüber kleinen Verschiebungen macht.
- **Flatten**: Wandelt die 2D-Feature-Maps in einen 1D-Vektor um.
Am Ende nutzen wir eine 'softmax'-Aktivierung, um Wahrscheinlichkeiten für jede Klasse (Hund, Blume oder Hintergrund) zu erhalten.

### Wichtige Konzepte für Studierende:
- **Normalisierung**: Wir teilen die Pixelwerte (0-255) durch 255, um sie auf den Bereich [0, 1] zu skalieren. Dies hilft dem neuronalen Netz, schneller zu lernen.
- **Train/Validation Split**: Wir behalten 20% der Daten zurück, um das Modell mit Bildern zu testen, die es während des Trainings nicht gesehen hat. Dies zeigt uns, ob das Modell "überfittet" (auswendig lernt) oder tatsächlich verallgemeinert.

Um das Training zu starten, führen Sie aus:
```bash
python src/custom_grid_env/cnn_tutorial/train.py
```

## 3. Bewertung der Ergebnisse

Nach dem Training speichert das Skript zwei Diagramme im Verzeichnis `results/`:
- **training_metrics.png**: Zeigt, wie sich Genauigkeit (Accuracy) und Fehlerwert (Loss) über die Zeit verändert haben. Idealerweise sollte die Genauigkeit steigen und der Fehler sinken.
- **confusion_matrix.png**: Zeigt, wo das Modell Fehler gemacht hat (z. B. wie viele Blumen fälschlicherweise als Hunde klassifiziert wurden).

## 4. Integration in die Umgebung

Das trainierte Modell wird als `model.keras` gespeichert. Der `PygameRenderer` in `src/custom_grid_env/renderer.py` sucht automatisch nach dieser Datei.

Der Renderer führt die Vorhersage in jedem Schritt aus:
1. Erstellt einen 64x64 Pixel "Schnappschuss" des aktuellen Feldes.
2. Übergibt diesen an das CNN.
3. Das CNN liefert Wahrscheinlichkeiten für alle drei Klassen (`dog`, `flower`, `background`).
4. Die Klasse mit der höchsten Wahrscheinlichkeit wird als Vorhersage im Info-Panel angezeigt.

---

### Übungen für Studierende:
1. **Daten anpassen**: Ändern Sie `line_spacing` in `data_generation.py` und beobachten Sie die Auswirkungen auf das Training.
2. **Modell experimentieren**: Füge eine weitere `Conv2D`-Schicht in `train.py` hinzu und vergleiche die Genauigkeit.
3. **Hyperparameter**: Ändern Sie die Anzahl der `epochs` oder die `batch_size` und analysieren Sie die Trainingskurven.
