import numpy as np
from custom_grid_env.particle_filter import ParticleFilter
from custom_grid_env.env import CustomGridEnv


def test_particle_filter_init():
    pf = ParticleFilter(rows=4, cols=5, num_particles=100)
    assert len(pf.particles) == 100
    assert len(pf.weights) == 100
    assert np.allclose(np.sum(pf.weights), 1.0)


def test_particle_filter_predict():
    env = CustomGridEnv()
    pf = ParticleFilter(rows=env.rows, cols=env.cols, num_particles=100)
    # Start all particles at [0, 2]
    pf.particles = [[0, 2]] * 100
    # Move right
    pf.predict(action=2, slip_prob=0.0, env_is_move_valid_fn=env._is_move_valid)
    for p in pf.particles:
        assert p == [0, 3]


def test_particle_filter_update_color():
    env = CustomGridEnv()
    pf = ParticleFilter(rows=env.rows, cols=env.cols, num_particles=2)
    pf.particles = [[0, 0], [0, 1]]  # [0,0] is Green (2), [0,1] is Red (1)
    pf.weights = np.array([0.5, 0.5])

    # Measurement is Green (2)
    pf.update({"color_measurement": 2}, "color", env.grid, [])
    # Particle at [0,0] matches, so its weight should be higher
    assert pf.weights[0] > pf.weights[1]
    assert np.allclose(np.sum(pf.weights), 1.0)


def test_particle_filter_resample():
    pf = ParticleFilter(rows=4, cols=5, num_particles=100)
    pf.weights = np.zeros(100)
    pf.weights[0] = 1.0  # Only the first particle has weight
    pf.particles[0] = [1, 1]

    pf.resample()
    for p in pf.particles:
        assert p == [1, 1]
