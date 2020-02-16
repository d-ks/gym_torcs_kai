# This is a driving example by a simple rule-based agent for the "Gym-TORCS-Kai" environment.

import numpy as np
import gym

# Make the environment.
env = gym.make("gym_torcs_kai:GymTorcsKai-v0")

# Set the number of dimensions of observation data.
env.obsdim = 31

# Initialize the environment and receive the initial state.
obs = env.reset()

# Set the terminal state flag (done) to False.
done = False

# Initialize the steer value.
steer = 0

# Repeat until the terminal state.
while not done:

    # Extract the states (angle, trackPos)
    # necessary for calculating the action from observation.
    angle = obs[0]
    trackPos = obs[25]

    # Calculate action value (steering): algorithm base on "drive_example" in SnakeOil.
    steer = angle * 10 / np.pi
    steer -= trackPos * 0.1

    # Execute the action in the environment,
    # and receive the information (next state, reward, terminal state flag)
    # from the environment.
    obs, reward, done, _ = env.step(np.array([steer]))

# Shut down the environment.
env.close()
