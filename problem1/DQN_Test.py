# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from agents.DQN_Agent import DQNAgent
from agents.Random_Agent import RandomAgent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# If you want to render the environment
env = gym.make('LunarLander-v3', render_mode = "human")

n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
agent = DQNAgent(n_actions=n_actions, dim_state=dim_state, epsilon_0=0.0, epsilon_inf=0.0)


agent.load_model('neural-network-1.pth')
#agent = RandomAgent(n_actions=n_actions)

done = False
state, info = env.reset()
total_reward = 0
while not done:
    action = agent.forward(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
print("Total reward:", total_reward)