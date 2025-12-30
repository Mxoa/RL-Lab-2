# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import os
from random import random
from time import time
from typing import OrderedDict
from importlib_metadata import files
import numpy as np
import gymnasium as gym
import torch
from tqdm import trange
import warnings, sys

from DQN_Agent import DQNAgent
warnings.simplefilter(action='ignore', category=FutureWarning)


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

TESTING_PATH = './models/underway/'
SOLVERS_PATH = './models/solvers/'
FAILERS_PATH = './models/failers/'

# Load model
try:
    ongoing_models_files = [file for file in os.listdir(TESTING_PATH) if file.endswith('.pth')]
    print('Ongoing models found:', ongoing_models_files)
    if len(ongoing_models_files) == 0:
        raise FileNotFoundError
    print('Loading model from file:', ongoing_models_files[0])
    model = torch.load(os.path.join(TESTING_PATH, ongoing_models_files[0]), weights_only=False)

    if not isinstance(model, torch.nn.Module):
        print('Loaded file is not a valid PyTorch model.')
        print('Converting to torch.nn.Module...')
        try:
            agent = DQNAgent(n_actions=4, dim_state=8)
            agent.load_model(os.path.join(TESTING_PATH, ongoing_models_files[0]))
            model = agent.policy_network
            print('Conversion successful.')
        except Exception as e:
            print('Conversion failed:', e)
            sys.exit(-1)
    print('Network model: {}'.format(model))
except FileNotFoundError:
    print('File ?.pth not found in ./models/underway/')
    sys.exit(-1)

# Import and initialize Mountain Car Environment
env = gym.make('LunarLander-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v3', render_mode = "human")


env.reset()

# Parameters
N_EPISODES = 150            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50

# Reward
episode_reward_list = []  # Used to store episodes reward

# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    while not (done or truncated):
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        q_values = model(torch.tensor(state))
        #print('q_values from model:', q_values)
        _, action = torch.max(q_values, dim=1)
        temp = env.step(action.item())
        next_state, reward, done, truncated, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)


# Close environment 
env.close()


avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))


timestamp = int(time())
model_name = f'neural-network-{timestamp}_{avg_reward:.1f}.pth'

if avg_reward - confidence >= CONFIDENCE_PASS:
    print('Your policy passed the test!')
    torch.save(model, os.path.join(SOLVERS_PATH, model_name))
else:
    print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))
    torch.save(model, os.path.join(FAILERS_PATH, model_name))

print('Model saved as {}'.format(model_name))

# rename the testing file to avoid overwriting
new_testing_name = f'neural-network-{timestamp}_{avg_reward:.1f}.tested'
os.rename(os.path.join(TESTING_PATH, ongoing_models_files[0]), os.path.join(TESTING_PATH, new_testing_name))