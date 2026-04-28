# Docstring Style Guide

In diesem Projekt verwenden wir den **Google Style** für Docstrings. Dies ermöglicht eine automatische Generierung der Dokumentation mit `mkdocstrings`.

## Beispiel

```python
def example_function(x: int, y: str = "default") -> bool:
    """Kurze Einzeilen-Beschreibung.

    Längere Beschreibung falls nötig. Kann mehrere Absätze
    umfassen.

    Args:
        x (int): Beschreibung des Parameters.
        y (str): Beschreibung. Defaults to "default".

    Returns:
        bool: Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Wann dieser Fehler auftritt.
    """
    return True
```

## Regeln

- Jede öffentliche Klasse und Methode **muss** einen Docstring haben.  
- Verwende `interrogate`, um die Abdeckung zu prüfen.  
- Dokumentiere alle Parameter und Rückgabewerte mit Typen.  
