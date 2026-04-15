# Tutorial: Partikelfilter und Sensorfusion

Dieses Tutorial erklärt, wie der Partikelfilter in der `CustomGrid`-Umgebung zur Lokalisierung des Agenten eingesetzt wird. Es richtet sich an Studierende der Informatik auf Bachelorniveau.

## Das Problem: Wo bin ich?

In einer realistischen Robotik-Anwendung kennt der Roboter seine exakte Position oft nicht. Er weiß zwar, welche Aktion er ausgeführt hat (z. B. "gehe einen Schritt vorwärts"), aber aufgrund von Unsicherheiten (Rutschen auf dem Boden) ist das Ergebnis stochastisch.

Der **Partikelfilter** ist ein Algorithmus, der eine Menge von Hypothesen (Partikeln) nutzt, um die wahrscheinliche Position des Agenten darzustellen.

## Funktionsweise des Partikelfilters

Ein Partikelfilter arbeitet in einem Zyklus aus drei Schritten:

1.  **Vorhersage (Prediction)**: Jedes Partikel wird entsprechend der Aktion des Agenten bewegt. Dabei wird das Bewegungsmodell (inklusive Rutschwahrscheinlichkeit) simuliert.
2.  **Korrektur (Update/Correction)**: Basierend auf den Sensormessungen wird jedes Partikel bewertet. Partikel, deren Position gut zu den Messungen passt, erhalten ein höheres Gewicht.
3.  **Resampling**: Partikel mit geringem Gewicht werden entfernt, während Partikel mit hohem Gewicht vervielfältigt werden. So konzentriert sich die "Cloud" auf die wahrscheinlichsten Orte.

## Sensorfusion in CustomGrid

Der Partikelfilter in dieser Umgebung kombiniert zwei verschiedene Sensortypen (**Sensorfusion**), um die Position zu schätzen:

### 1. Der Farbsensor
Der Agent hat einen Sensor, der die Bodenfarbe (Weiß, Rot, Grün) misst. Dieser Sensor hat eine Genauigkeit von **80%**.  
- Wenn ein Partikel auf einer roten Zelle liegt und der Sensor "Rot" meldet, steigt die Wahrscheinlichkeit für dieses Partikel.  
- Meldet der Sensor "Grün", obwohl das Partikel auf einer roten Zelle liegt, sinkt die Wahrscheinlichkeit.

### 2. Das CNN (Visuelle Erkennung)
Das trainierte Convolutional Neural Network liefert Wahrscheinlichkeiten für die Klassen `dog`, `flower` und `background`.
Der Partikelfilter nutzt diese Vorhersagen als Messwerte:  
- Jedes Partikel "schaut" in die Karte: Welches Objekt befindet sich an meiner (hypothetischen) Position?  
- Die Likelihood eines Partikels berechnet sich aus der Wahrscheinlichkeit, die das CNN für genau dieses Objekt ausgegeben hat.

## Mathematische Kombination

Wir nehmen an, dass die Sensoren bedingt unabhängig sind. Die Gesamtwahrscheinlichkeit $P$ für ein Partikel ergibt sich aus dem Produkt der Einzelwahrscheinlichkeiten:

$$p(z_{\text{color}}, z_{\text{cnn}} | s) = p(z_{\text{color}} | s) \cdot p(z_{\text{cnn}} | s)$$

Durch diese Kombination kann der Agent seine Position auch dann bestimmen, wenn ein einzelner Sensor sehr verrauscht ist. Wenn z.B. das CNN unsicher ist, kann der Farbsensor oft helfen, die Position auf dem Gitter einzugrenzen.

## Übung für Studierende

1.  **Einfluss der Sensoren**: Teste in der `Colab_GUI_Demo` den Partikelfilter nur mit dem Farbsensor, nur mit dem CNN und mit beiden. Beobachte, wie schnell die Partikelwolke konvergiert.
2.  **Rutschmodelle**: Vergleiche das "perpendicular" Rutschen mit dem "longitudinal" Rutschen. Welches Modell macht die Lokalisierung schwieriger?
3.  **Partikelanzahl**: Reduziere die Anzahl der Partikel im `AgentInterface`. Ab welcher Anzahl wird die Schätzung instabil?

### Interaktive Demo

Testen Sie den Partikelfilter direkt in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/Colab_GUI_Demo.ipynb)
