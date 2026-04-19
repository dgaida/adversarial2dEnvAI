import pytest
import numpy as np
from custom_grid_env.env import CustomGridEnv
from custom_grid_env.planner import TaskPlanner

def test_planner_initialization():
    env = CustomGridEnv(deterministic=True, use_ghost=False)
    planner = TaskPlanner(env)
    assert planner.env == env

def test_value_iteration():
    env = CustomGridEnv(deterministic=True, use_ghost=False)
    planner = TaskPlanner(env)
    goal = (3, 1) # Goal in the grid
    V = planner.value_iteration(goal)
    assert V.shape == (4, 5)
    assert V[3, 1] == 100.0 # Goal value is 0 in the iteration loop because we skip it
    # Neighbors should have high value
    assert V[3, 0] > 0
    assert V[2, 1] > 0

def test_get_optimal_action():
    env = CustomGridEnv(deterministic=True, use_ghost=False)
    planner = TaskPlanner(env)
    goal = (3, 1)
    V = planner.value_iteration(goal)
    # At (3, 0), move right (2) to get to (3, 1)
    action = planner.get_optimal_action((3, 0), V)
    assert action == 2

def test_solve_tsp():
    env = CustomGridEnv(deterministic=True, use_ghost=False)
    planner = TaskPlanner(env)
    start = (0, 2)
    targets = [(0, 0), (3, 4)]
    order = planner.solve_tsp(start, targets)
    assert len(order) == 2
    # One of the two possible orders
    assert set(order) == set(targets)

def test_get_path():
    env = CustomGridEnv(deterministic=True, use_ghost=False)
    planner = TaskPlanner(env)
    start = (3, 0)
    goal = (3, 1)
    path = planner.get_path(start, goal)
    assert path == [2] # Right
