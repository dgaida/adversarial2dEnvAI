# Grid Environment

The CustomGrid is a 4x5 grid with special fields, items, and obstacles.

## Grid Layout

The grid consists of 4 rows (0-3) and 5 columns (0-4).

## Cell Colours

Some cells have background colours that the agent can perceive:  
- **Red**: Used for certain patterns.  
- **Green**: Used for certain patterns.  
- **White**: Default colour.  

## Items

Various items are distributed throughout the world:  
- **Dog**: Classified by a CNN when the agent enters the cell.  
- **Flower**: Classified by a CNN when the agent enters the cell.  
- **Notes**: Displayed in the observations.  

## Walls

Walls block movement between cells.  
- **Horizontal Walls**: Block Up/Down movements.  
- **Vertical Walls**: Block Left/Right movements.  

## Task Planning

The environment is tightly integrated with the `TaskPlanner` to enable complex missions. The `TaskPlanner` utilizes:  
- **Grid Descriptions**: Automatically generated texts about the world state.  
- **LLM Integration**: Extraction of targets from natural language.  
- **Optimization**: TSP solver for efficient routes and Value Iteration for pathfinding.  

For more details, see the [Task Planning Tutorial](../tutorial/task_planning.md).
