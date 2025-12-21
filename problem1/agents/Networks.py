import torch

class DuelingNetwork(torch.nn.Module):
    def __init__(self, dim_state: int, n_actions: int, n = 128):
        super(DuelingNetwork, self).__init__()
        self.pre_net = torch.nn.Sequential(
            torch.nn.Linear(dim_state, n),
            torch.nn.ReLU()
        )
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(n, n),
            torch.nn.ReLU(),
            torch.nn.Linear(n, 1)
        )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(n, n),
            torch.nn.ReLU(),
            torch.nn.Linear(n, n_actions)
        )
        
    def forward(self, x):
        pre_net_result = self.pre_net(x)
        values = self.value_stream(pre_net_result)
        advantages = self.advantage_stream(pre_net_result)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values