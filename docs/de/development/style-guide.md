# Docstring Style Guide

Dieses Projekt folgt dem **Google Python Style Guide** für Docstrings.

## Grundstruktur

Jeder Docstring sollte eine kurze Zusammenfassung (einzeilig) enthalten, gefolgt von einer detaillierteren Beschreibung, falls nötig.

### Funktionen und Methoden

```python
def meine_funktion(parameter1: int, parameter2: str) -> bool:
    """
    Kurze Zusammenfassung der Funktion.

    Längere Beschreibung, die das Verhalten im Detail erklärt.

    Args:
        parameter1 (int): Beschreibung des ersten Parameters.
        parameter2 (str): Beschreibung des zweiten Parameters.

    Returns:
        bool: Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Wenn ein ungültiger Wert übergeben wird.
    """
```

### Klassen

```python
class MeineKlasse:
    """
    Kurze Zusammenfassung der Klasse.

    Längere Beschreibung der Klasse und ihrer Verantwortung.

    Attributes:
        attribut1 (int): Beschreibung des ersten Attributs.
    """
```

## Warum Google-Stil?

Der Google-Stil ist hochgradig lesbar (sowohl im Quellcode als auch generiert) und wird von `mkdocstrings` hervorragend unterstützt.
