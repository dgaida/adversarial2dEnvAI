from custom_grid_env.env import CustomGridEnv


def test_set_goal():
    env = CustomGridEnv()
    new_goal = (0, 0)
    env.set_goal(new_goal)
    assert env.grid[new_goal[0], new_goal[1]]["is_goal"] is True
    # Previous goal (3,1) should be False
    assert env.grid[3, 1]["is_goal"] is False


def test_get_grid_description_mapping():
    env = CustomGridEnv()
    desc = env.get_grid_description()
    # Check for mapping of one_note and two_notes
    assert "klassische Musik und Klaviermusik" in desc
    assert "Rockmusik" in desc
    # Check that Startpunkt/Zielpunkt are removed
    assert "Startpunkt" not in desc
    assert "Zielpunkt" not in desc
