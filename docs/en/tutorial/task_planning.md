# Task Planning & LLM Integration

This tutorial describes how the CustomGrid environment uses natural language to understand complex tasks, identify targets, and navigate to them efficiently.

## Workflow Overview

The task planning process consists of three main steps:
1. **Target Identification**  : An LLM (Large Language Model) extracts coordinates from a natural language description.
2. **Sequence Optimization**  : A TSP (Traveling Salesperson Problem) solver determines the shortest route to visit all targets.
3. **Path Planning**  : Value Iteration calculates the optimal moves to reach each target sequentially.

## 1. Target Identification with LLMs

The `TaskPlanner` class uses an LLM (defaulting via Groq) to interpret user instructions such as *"Visit the dog first, then the flowers"*.

### Grid Description
To let the LLM know where objects are located, the environment generates a text description of the current grid (`get_grid_description()`). Items are mapped as follows:
- **dog**: "ein Bild eines Hundes" (an image of a dog)
- **flower**: "eine Blume" (a flower)
- **two_flowers**: "zwei Blumen" (two flowers)

### Prompting & Extraction
The LLM receives a system prompt instructing it to return **only** a JSON array of coordinates (e.g., `[[0, 1], [2, 3]]`).

Since modern models often output "reasoning" in `<think>` tags, the `TaskPlanner` includes robust cleaning logic:
- Removing `<think>` blocks using regex.
- Extracting the JSON array from the remaining text.
- Validating the format to avoid parsing errors.

## 2. Route Optimization (TSP)

Once the target coordinates are identified, the most efficient visiting order must be found.

### Distance Matrix
The `TaskPlanner` first computes a distance matrix between the starting point and all targets. Since the environment contains walls, a Breadth-First Search (BFS) is used to determine the actual shortest path distance.

### TSP Solver
The solver tries all permutations of the target sequence and selects the one with the lowest total distance (including returning to the starting point). Given the small number of targets in this scenario, this brute-force approach is extremely fast and guarantees global optimality.

## 3. Path Planning via Value Iteration

To reach a specific target in the grid, **Value Iteration** is employed.

- **State Values**: Each cell receives a value based on its distance to the target. The target itself has the highest value (+100).
- **Iterative Updates**: Values are updated until they converge (`theta < 0.0001`).
- **Action Selection**: In each step, the agent chooses the action leading to the adjacent cell with the highest value.

This approach is robust to the environment's walls and guarantees the shortest path to the target.

## Integration in Colab GUI

In the `ColabGUI`, this process is split into two steps:
1. **Plan**  : The LLM is called, targets are visualized, and the optimal tour is displayed as a checklist.
2. **Execute**  : The agent works through the list of targets one by one, with progress marked in real-time in the GUI.

### Exercise for Students
1. **Ambiguity**  : Enter an unclear instruction (e.g., "Go to something beautiful"). How does the LLM react based on the grid description?
2. **Complexity**  : Create a task with 4-5 targets. Observe how the TSP solver changes the sequence compared to the order of mention in the text.
3. **Obstacles**  : Place walls so that a direct path is blocked. Verify that Value Iteration still finds the optimal detour.
