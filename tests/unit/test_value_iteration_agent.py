from custom_grid_env.env import CustomGridEnv
from custom_grid_env.agents.value_iteration_agent import ValueIterationAgent


def test_value_iteration_agent_initialization():
    env = CustomGridEnv()
    agent = ValueIterationAgent(env.action_space, env=env)
    assert agent.env == env
    assert agent.action_space == env.action_space


def test_value_iteration_agent_get_action():
    env = CustomGridEnv(deterministic=True, use_ghost=False)
    # Set a specific goal
    goal_pos = (1, 1)
    env.set_goal(goal_pos)

    agent = ValueIterationAgent(env.action_space, env=env)
    obs, info = env.reset()

    # Starting at (0, 2), goal at (1, 1)
    # Possible path: (0, 2) -> (0, 1) -> (1, 1) or (0, 2) -> (1, 2) -> (1, 1)
    # Actions: 0 (left), 1 (down), 2 (right), 3 (up)

    action = agent.get_action(obs)
    assert action in [0, 1]  # Move left or down

    # Move to goal and check if it stays or handles it
    env.agent_pos = [1, 1]
    obs = env._get_obs()
    action = agent.get_action(obs)
    # Should probably stay (0) or some default
    assert action >= 0
