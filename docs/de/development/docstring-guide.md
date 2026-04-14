# Docstring Standard

Wir verwenden den **Google Style Guide** für alle Docstrings im Projekt. Dies gewährleistet eine konsistente API-Dokumentation.

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

    Example:
        >>> example_function(42)
        True
    """
    return True
```

## Abdeckung

Die Einhaltung wird automatisch mit `interrogate` geprüft. Eine Abdeckung von mindestens 95% ist erforderlich.
