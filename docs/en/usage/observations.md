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

## Info Dictionary

In addition to the observations, the environment returns an `info` dictionary that may contain the following additional information:

- `cnn_prediction`: A tuple `(class_name, probability)` if the agent is on a cell with a dog or flower and a trained model is loaded.  
- `color_measurement`: A noisy measurement of the ground color (0=white, 1=red, 2=green). The sensor is 80% accurate.  
- `intended_action`: The action intended by the agent.  
- `actual_action`: The action actually performed (may differ if slipping occurs).  
