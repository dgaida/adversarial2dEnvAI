# Docstring Style Guide

This project follows the **Google Python Style Guide** for docstrings.

## Basic Structure

Each docstring should contain a short summary (one line), followed by a more detailed description if necessary.

### Functions and Methods

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Short summary of the function.

    Longer description explaining the behavior in detail.

    Args:
        param1 (int): Description of the first parameter.
        param2 (str): Description of the second parameter.

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: If an invalid value is passed.
    """
```

### Classes

```python
class MyClass:
    """
    Short summary of the class.

    Longer description of the class and its responsibility.

    Attributes:
        attr1 (int): Description of the first attribute.
    """
```

## Why Google Style?

The Google style is highly readable (both in source code and generated) and is excellently supported by `mkdocstrings`.
