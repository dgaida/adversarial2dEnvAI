from custom_grid_env.interface import AgentInterface
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent


def test_interface_loop():
    interface = AgentInterface(render=False, slip_probability=0.0)
    obs = interface.reset(seed=42)
    agent = RandomPlayerAgent(interface.get_action_space())

    # Run a few steps
    for _ in range(5):
        if interface.is_terminated():
            break
        action = agent.get_action(obs)
        obs, reward, done, info = interface.step(action)

    stats = interface.get_episode_stats()
    assert stats["steps"] > 0
    interface.close()


def test_interface_termination():
    interface = AgentInterface(render=False, slip_probability=0.0)
    interface.reset()

    # Force termination by setting agent pos to goal
    interface.env.agent_pos = [3, 1]
    # Any action should trigger goal reach
    obs, reward, done, info = interface.step(2)

    assert done is True
    assert interface.is_terminated() is True
    assert interface.get_episode_stats()["reached_goal"] is True
    interface.close()
