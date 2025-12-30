# Claire Binet-Tarbé de Vauxclairs (20020701-T842)
# Marceau Messaussier (20030805-T390)

# Problem 1: Deep Q-Network (DQN) for Lunar Lander

This directory contains the implementation and analysis tools for training and evaluating a DQN agent on the Lunar Lander environment.

## Main Files

### `problem_1.py`
Main training script for the DQN agent. This script:
- Loads configuration parameters from `config.yaml`
- Trains a DQN agent on the Lunar Lander environment
- Saves the trained model at the end of training in `models/underway/`
- Automatically saves models during training if the average reward over the last 50 episodes is ≥ 50
- Generates training plots and saves experimental data to CSV files in `data_experiments/`

### `manager.py`
Script for managing and comparing multiple training experiments. This script:
- Runs multiple training scenarios with different hyperparameters (discount factor, replay buffer size, number of episodes)
- Generates comparison plots showing the impact of different hyperparameters
- Creates side-by-side plots for average reward and average number of steps per episode
- Saves plots in the `plots/` directory with LaTeX-friendly formatting
- Uses existing CSV data from previous training runs

### `plotter.py`
Visualization tool for analyzing trained DQN models. This script:
- Generates 3D surface plots of the learned Q-value function
- Visualizes the optimal policy by plotting the action selection over state space
- Creates two types of plots:
  - **Value plot**: Maximum Q-values across the state space (y, omega)
  - **Action plot**: Optimal action selection across the state space
- Saves plots to `plots/value_plot.png` and `plots/action_plot.png`

## Agent Implementations

### `Agent.py`
Base agent class that provides the interface for all agents. Defines the common methods: `forward()`, `backward()`, `observe()`, and `describe()`.

### `DQN_Agent.py`
Deep Q-Network agent implementation. This class:
- Implements the DQN algorithm with experience replay
- Supports epsilon-greedy exploration policy
- Includes optional features: Combined Experience Replay (CER) and Dueling Network Architecture
- Provides methods for saving and loading trained models

### `Random_Agent.py`
Random agent implementation used as a baseline. Selects actions uniformly at random from the available action space.

### `Networks.py`
Neural network architectures used by the DQN agent, including the Dueling Network Architecture implementation.

## Supporting Files

### `DQN_check_solution.py`
Evaluation script provided in the assignment. Tests a trained model over 50 episodes and reports whether it passes the performance threshold (average reward ≥ 50 with 95% confidence).

### `DQN_Test.py`
Interactive visualization tool that allows you to watch a trained agent perform in the Lunar Lander environment with rendering enabled.

### `DQN_check_solution_random_agent.py`
Baseline evaluation script that tests a random agent to establish the baseline performance (used for question a).

### `DQN_check_solution_in_forlder_underway.py`
Automated testing script that:
- Finds untested models in `models/underway/`
- Evaluates their performance
- Moves them to `models/solvers/` if they pass the test, or `models/failers/` if they fail

## Configuration

All training parameters are configured in `config.yaml`, including:
- Number of episodes
- Discount factor (gamma)
- Learning rate
- Epsilon-greedy exploration parameters
- Batch size and replay buffer size
- Target network update frequency
- Gradient clipping value
- Use of Combined Experience Replay (CER) and Dueling Network Architecture

## Directory Structure

```
problem1/
├── Agent.py            # Base agent class
├── DQN_Agent.py        # DQN agent implementation
├── Random_Agent.py      # Random agent implementation
├── Networks.py         # Neural network architectures
├── problem_1.py        # Main training script
├── manager.py          # Experiment management script
├── plotter.py          # Visualization tools
├── models/             # Saved models
│   ├── solvers/        # Models that passed the test
│   ├── failers/       # Models that failed the test
│   └── underway/      # Models currently being tested
├── data_experiments/   # CSV files with training data
├── plots/              # Generated plots and visualizations
├── neural-network-1.pth  # Trained Q-network for problem 1
└── config.yaml        # Configuration file
```


