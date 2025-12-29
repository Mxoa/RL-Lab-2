# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
from time import time
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from agents.DQN_Agent import DQNAgent
from agents.Random_Agent import RandomAgent
import warnings
import argparse
import yaml
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(
    "ignore", 
    message="pkg_resources is deprecated as an API", 
    category=UserWarning
)

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

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v3', render_mode = "human")


# Lecture de la configuration depuis le fichier YAML


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file) # type de config : dict


env.reset()

parser = argparse.ArgumentParser(description="DQN Lunar Lander training parameters")
parser.add_argument('--episodes', '-e', type=int, default=config['episodes'], help='Number of episodes')
parser.add_argument('--running-average', '-r', type=int, default=config['running_average'], help='Window size for running average')
parser.add_argument('--discount-factor', '-g', type=float, default=config['discount_factor'], help='Discount factor (gamma)')
parser.add_argument('--learning-rate', '-l', type=float, default=config['learning_rate'], help='Learning rate')
parser.add_argument('--epsilon-0', type=float, default=config['epsilon_0'], help='Initial epsilon (epsilon_0)')
parser.add_argument('--epsilon-inf', type=float, default=config['epsilon_inf'], help='Minimum epsilon (epsilon_inf)')
parser.add_argument('--epsilon-decay', type=float, default=config['epsilon_decay'], help='Epsilon decay rate')
parser.add_argument('--batch-size', '-b', type=int, default=config['batch_size'], help='Minibatch size')
parser.add_argument('--replay-buffer-size', type=int, default=config['replay_buffer_size'], help='Replay buffer capacity')
parser.add_argument('--target-update-freq', type=int, default=config['target_update_freq'], help='Target network update frequency (in steps)')
parser.add_argument('--cutting-value', type=float, default=config['cutting_value'], help='Gradient clipping / cutting value')
parser.add_argument('--show-plot', action='store_true', default=config['show_plot'], help='Show training plot in real-time')
parser.add_argument('--cer' , action='store_true', default=config['use_cer'], help='Use Combined Experience Replay (CER)')
parser.add_argument('--dueling', action='store_true', default=config['dueling'], help='Use Dueling Network Architecture')
args, _ = parser.parse_known_args()

N_episodes = args.episodes
n_ep_running_average = args.running_average
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

SHOW_PLOT = args.show_plot

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = DQNAgent(
    n_actions,
    dim_state,
    discount_factor=args.discount_factor,
    learning_rate=args.learning_rate,
    epsilon_0=args.epsilon_0,
    epsilon_inf=args.epsilon_inf,
    epsilon_decay=args.epsilon_decay,
    batch_size=args.batch_size,
    replay_buffer_size=args.replay_buffer_size,
    target_update_freq=args.target_update_freq,
    cutting_value=args.cutting_value,
    use_cer=args.cer,
    dueling=args.dueling
)

#agent = RandomAgent(n_actions)

agent.describe()

# Load pre-trained model weights (if any)
# agent.load_model('models/solvers/neural-network-1765459890_121.4.pth')

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

plt.figure()
plt.title("DQN Training Loss Curve")
plt.ion()

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    max_avg = 50.0
    while not (done or truncated):
        # Take a random action
        action = agent.forward(state)

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)

        # Observe reward
        agent.observe(state, action, reward, next_state, done)

        # Update episode reward
        total_episode_reward += reward

        # Update the agent's (Q-value, policy, etc.)
        loss = agent.backward()

        # Update state for next iteration
        state = next_state
        t+= 1
    
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    if i > 50 and SHOW_PLOT:
        plt.pause(0.001)
        plt.plot(agent.step_count, running_average(episode_reward_list, n_ep_running_average)[-1], color='blue', marker='o')


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - eps {:.3f}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1],
        agent.epsilon
        ))
    avg = running_average(episode_reward_list, n_ep_running_average)[-1]
    # Save the trained model
    if avg >= max_avg and episode_reward_list[-1] >= 200.0:
        print("\nEnvironment solved in {} episodes!".format(i))
        max_avg = avg
        agent.save_model(f'models/underway/success_whole_model_{int(time())}.pth', save_whole_model=True)
    
plt.ioff()
if args.show_plot:
    plt.show()


# Save the trained model
agent.save_model(f'models/underway/whole_model_{int(time())}_gamma_{args.discount_factor}.pth', save_whole_model=True)

# Close environment
env.close()

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)

timestamp = int(time())
plt.savefig(f'plots/results_gamma_{args.discount_factor}_buffer_{args.replay_buffer_size}_{timestamp}.png')
print(f"Graphique sauvegardé sous le nom : results_gamma_{args.discount_factor}...")

if SHOW_PLOT:
    plt.show()
plt.close()

import pandas as pd
import os

# On crée un nom de fichier basé sur les paramètres clés
filename = f"results_gamma{args.discount_factor}_mem{args.replay_buffer_size}_eps{args.episodes}.csv"

# Création du dossier 'data' si inexistant
if not os.path.exists("data_experiments"):
    os.makedirs("data_experiments")

path = os.path.join("data_experiments", filename)

df = pd.DataFrame({
    'episode': range(len(episode_reward_list)),
    'reward': episode_reward_list,
    'running_avg': running_average(episode_reward_list, n_ep_running_average),
    'steps': episode_number_of_steps
})

df.to_csv(path, index=False)
print(f"Données sauvegardées dans : {path}")

# On désactive le plt.show() bloquant si on n'a pas demandé explicitement l'affichage
if SHOW_PLOT:
    plt.show()
plt.close()