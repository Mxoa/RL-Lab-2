from agents.Agent import *
from agents.Networks import DuelingNetwork


class DQNAgent(Agent):

    ''' Deep Q-Network Agent, child of the class Agent

        Args:
            n_actions (int): number of actions
            dim_state (int): state dimensionality
            discount_factor (float): discount factor for future rewards
            learning_rate (float): learning rate for the optimizer
            epsilon_0 (float): initial value of epsilon for the epsilon-greedy policy
            epsilon_inf (float): minimum value of epsilon
            epsilon_decay (float): decay rate of epsilon per step
            batch_size (int): size of the batches sampled from the replay buffer
            replay_buffer_size (int): maximum size of the replay buffer
            target_update_freq (int): frequency (in steps) to update the target network

        Attributes:
            n_actions (int): where we store the number of actions
            dim_state (int): where we store the state dimensionality
            discount_factor (float): where we store the discount factor for future rewards
            learning_rate (float): where we store the learning rate for the optimizer
            batch_size (int): where we store the size of the batches sampled from the replay buffer
            replay_buffer_size (int): where we store the maximum size of the replay buffer
            target_update_freq (int): where we store the frequency (in steps) to update the target network
            replay_buffer (list): list to store the experience tuples
            step_count (int): counter to keep track of the number of steps taken
            policy_network (torch.nn.Module): neural network for approximating the Q-function
            target_network (torch.nn.Module): target neural network for stable Q-value updates
            optimizer (torch.optim.Optimizer): optimizer for training the policy network
            loss_fn (torch.nn.Module): loss function for training the policy network
            use_cer (bool): whether to use Combined Experience Replay (CER)
            dueling (bool): whether to use Dueling Network Architecture

    '''

    def __init__(self,
                 n_actions: int,
                 dim_state: int,
                 discount_factor: float = 0.85,
                 learning_rate: float = 2e-4,
                 epsilon_0: float = 0.99,
                 epsilon_inf: float = 0.05,
                 epsilon_decay: float = 0.999,
                 batch_size: int = 64,
                 replay_buffer_size: int = 20000,
                 target_update_freq: int = 2000,
                 cutting_value: float = 2.0,
                 use_cer: bool = True,
                 dueling: bool = False):
        super(DQNAgent, self).__init__(n_actions)
        self.dim_state = dim_state
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon_0 = epsilon_0
        self.epsilon_inf = epsilon_inf
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_0
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_freq = target_update_freq
        self.CER = use_cer
        self.dueling = dueling

        self.replay_buffer = []
        self.step_count = 0
        self.CUTTING_VALUE = cutting_value

        self.policy_network = self.__build_network()
        self.target_network = self.__build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def __build_network(self):
        ''' Build the neural network for approximating the Q-function

            Returns:
                model (torch.nn.Module): the neural network model
        '''
        if self.dueling:
            model = DuelingNetwork(self.dim_state, self.n_actions, 128)
            return model
        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(self.dim_state, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.n_actions)
            )
            return model
    
    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action using an epsilon-greedy policy

            Args:
                state (np.ndarray): current state

            Returns:
                action (int): the selected action
        '''
        self.epsilon = max(self.epsilon_inf, self.epsilon * self.epsilon_decay)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_network(state_tensor)
            _, action = torch.max(q_values, dim=1)
            return action.item()
        
    def observe(self, state, action, reward, next_state, done):
        ''' Store the experience tuple in the replay buffer '''
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def backward(self):

        ''' Update the policy network using a batch of experiences from the replay buffer '''
        
        if len(self.replay_buffer) < self.batch_size:
            return

        
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])

        if self.CER:
            states = list(states)
            actions = list(actions)
            rewards = list(rewards)
            next_states = list(next_states)
            dones = list(dones)

            last_experience = self.replay_buffer[-1]
            states[-1] = last_experience[0]
            actions[-1] = last_experience[1]
            rewards[-1] = last_experience[2]
            next_states[-1] = last_experience[3]
            dones[-1] = last_experience[4]

        #print("Batch sampled from replay buffer. with size:", len(batch) , "Total buffer size:", len(self.replay_buffer))

        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        q_next = self.target_network(next_states_tensor).max(1)[0].detach().numpy()
        y = []
        for i in range(self.batch_size):
            if dones[i]:
                y.append(rewards[i])
            else:
                y.append(rewards[i] + self.discount_factor * q_next[i])

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        q_values = self.policy_network(states_tensor).gather(1, actions_tensor)
        loss = self.loss_fn(q_values, y_tensor)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.CUTTING_VALUE)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        #print("Loss:", loss.item())

        return loss.item()

    def save_model(self, filepath: str, save_whole_model: bool = True):
        ''' Save the policy network model to a file

            Args:
                filepath (str): path to the file where the model will be saved
        '''
        # torch.save(your neural network, ’neural-network-1.pt’).
        if save_whole_model:
            torch.save(self.policy_network, filepath)
        else:
            torch.save(self.policy_network.state_dict(), filepath)

    def load_model(self, filepath: str, load_whole_model: bool = True):
        ''' Load the policy network model from a file

            Args:
                filepath (str): path to the file from which the model will be loaded
        '''
        if load_whole_model:
            self.policy_network = torch.load(filepath, weights_only=False)
            self.target_network = torch.load(filepath, weights_only=False)
        else:
            self.policy_network.load_state_dict(torch.load(filepath))
            self.target_network.load_state_dict(self.policy_network.state_dict())