# Installation

In diesem Abschnitt wird erklärt, wie du die `custom_grid_env` Bibliothek auf deinem System installierst.

## Voraussetzungen

Stelle sicher, dass du Python 3.8 oder neuer installiert hast. Es wird empfohlen, eine virtuelle Umgebung zu verwenden.

## Installation vom Quellcode

1. Klone das Repository:  
   ```bash
   git clone https://github.com/dgaida/adversarial2dEnvAI.git
   cd adversarial2dEnvAI
   ```

2. Installiere das Paket im editierbaren Modus:  
   ```bash
   pip install -e .
   ```

Dies installiert alle notwendigen Abhängigkeiten wie `gymnasium`, `numpy`, `pygame`, `tensorflow`, `matplotlib` und `scikit-learn`.

## Installation mit Anaconda

Wenn du Anaconda oder Miniconda verwendest, kannst du die Umgebung direkt aus der `environment.yml` Datei erstellen:

```bash
# Erstelle und aktiviere die Umgebung
conda env create -f environment.yml
conda activate custom_grid_env

# Installiere das Paket im editierbaren Modus
pip install -e .
```

## Verifizierung

Du kannst die Installation überprüfen, indem du versuchst, das Paket in Python zu importieren:

```python
import custom_grid_env
print(custom_grid_env.__version__)
```
