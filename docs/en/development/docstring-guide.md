# Docstring Style Guide

In this project, we use the **Google Style** for docstrings. This allows for automatic documentation generation with `mkdocstrings`.

## Example

```python
def example_function(x: int, y: str = "default") -> bool:
    """Short one-line description.

    Longer description if necessary. Can span multiple paragraphs.

    Args:
        x (int): Description of the parameter.
        y (str): Description. Defaults to "default".

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: When this error occurs.
    """
    return True
```

## Rules

- Every public class and method **must** have a docstring.  
- Use `interrogate` to check coverage.  
- Document all parameters and return values with types.  
