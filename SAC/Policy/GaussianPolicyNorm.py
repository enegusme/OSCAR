import torch
import torch.nn as nn

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 act_min: torch.Tensor, act_max: torch.Tensor, std_min: float, std_max: float):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.mean_net = nn.Linear(hidden_dim, act_dim)
        self.stddev_net = nn.Linear(hidden_dim, act_dim)

        self.action_scale = (act_max - act_min) / 2
        self.action_bias = (act_max + act_min) / 2

        self.std_min = std_min
        self.std_max = std_max

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_net(state)
        mean = self.mean_net(shared)
        stddev = self.stddev_net(shared)
        stddev = torch.tanh(stddev)
        stddev = self.std_min + 0.5 * (self.std_max - self.std_min) * (stddev + 1)
        return mean, stddev

    def get_action(self, state, deterministic=False, with_log_prob=False) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, stddev = self(state)
        stddev = stddev.exp()
        if not deterministic:
            normal = torch.distributions.Normal(mean, stddev)
            sample = normal.rsample()
            tan_sample = torch.tanh(sample)
            action = tan_sample * self.action_scale + self.action_bias
            if with_log_prob:
                log_prob = normal.log_prob(sample)
                log_prob -= torch.log(self.action_scale * (1 - tan_sample.pow(2)) + 1e-6)
                log_prob = log_prob.sum(1, keepdim=True)
            else:
                log_prob=None
        else:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            log_prob = None
        return action, log_prob
