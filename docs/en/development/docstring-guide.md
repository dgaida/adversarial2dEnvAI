# Docstring Standard

We use the **Google Style Guide** for all docstrings in the project. This ensures consistent API documentation.

## Example

```python
def example_function(x: int, y: str = "default") -> bool:
    """Short one-line description.

    Longer description if necessary. Can include multiple
    paragraphs.

    Args:
        x (int): Description of the parameter.
        y (str): Description. Defaults to "default".

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: When this error occurs.

    Example:
        >>> example_function(42)
        True
    """
    return True
```

## Coverage

Compliance is automatically checked with `interrogate`. A coverage of at least 95% is required.
