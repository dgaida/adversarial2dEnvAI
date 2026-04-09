from custom_grid_env.env import CustomGridEnv


def test_env_init():
    env = CustomGridEnv(render_mode="rgb_array")
    assert env.rows == 4
    assert env.cols == 5
    assert env.render_mode == "rgb_array"
    env.close()


def test_env_reset():
    env = CustomGridEnv()
    obs, info = env.reset(seed=42)
    assert "current_cell" in obs
    assert "neighbors" in obs
    assert "ghost_relative_pos" in obs
    assert info["current_turn"] == "agent"
    assert env.agent_pos == [0, 2]
    env.close()


def test_env_step_agent():
    env = CustomGridEnv(slip_probability=0.0)
    env.reset()
    # Initial pos [0,2]. Action 1 is DOWN. [1,2] is valid.
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.agent_pos == [1, 2]
    assert env.current_turn == 1  # Now ghost's turn
    assert info["mover"] == "agent"
    env.close()


def test_env_step_ghost():
    env = CustomGridEnv(slip_probability=0.0)
    env.reset()
    # Agent moves first
    env.step(3)
    # Ghost's turn. Initial ghost pos [0,3]. Action 0 is LEFT.
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.ghost_pos == [0, 2]
    assert env.current_turn == 0  # Now agent's turn
    assert info["mover"] == "ghost"
    env.close()


def test_env_wall_collision():
    env = CustomGridEnv(slip_probability=0.0)
    env.reset()
    # vertical[1,2] is True, meaning wall between col 2 and 3 at row 1.
    # Move agent to [1,2].
    env.agent_pos = [1, 2]
    # Try to move right to [1,3]. Action 2 is RIGHT.
    env.step(2)
    assert env.agent_pos == [1, 2]  # Should be blocked
    env.close()


def test_env_goal_reached():
    env = CustomGridEnv(slip_probability=0.0)
    env.reset()
    # Goal at [3,1]
    env.agent_pos = [2, 1]
    # Move down to [3,1]. Action 1 is DOWN.
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.agent_pos == [3, 1]
    assert terminated is True
    assert reward == 100.0
    assert info.get("reached_goal") is True
    env.close()


def test_env_caught_by_ghost():
    env = CustomGridEnv(slip_probability=0.0)
    env.reset()
    # Agent at [2,3], ghost at [1,3]. horizontal[1,3] is True (between row 1 and 2).
    # Wait, horizontal[1,3] is True. So row 1 and 2 are blocked.
    # Let's use [0,0] and [0,1]. vertical[0,0] is True (between col 0 and 1).
    # Let's use [3,3] and [3,4]. No walls there.
    env.agent_pos = [3, 4]
    env.ghost_pos = [3, 3]
    # Agent moves left to [3,3]. Action 0 is LEFT.
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.agent_pos == [3, 3]
    assert terminated is True
    assert reward == -50.0
    assert info.get("caught_by_ghost") is True
    env.close()
