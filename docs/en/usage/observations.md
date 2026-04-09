# Observations

## Structure

The observation is a nested dictionary:

- `current_cell`: Info about the current cell (colour, items, goal status, text).
- `neighbors`: Accessibility and colour of the 4 neighboring cells.
- `ghost_relative_pos`: Relative position of the ghost to the agent `[row_diff, col_diff]`.

## Details

| Field | Type | Description |
|-------|------|-------------|
| `colour` | int | 0=white, 1=red, 2=green |
| `is_goal` | int | 1 if goal cell, else 0 |
| `accessible` | int | 1 if traversable, 0 if blocked by wall or boundary |
