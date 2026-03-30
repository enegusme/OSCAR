import numpy as np
import torch
from SAC.Policy.GaussianPolicyNorm import GaussianPolicy
from SAC.Networks.QNetworkNorm import QNetwork

class SAC:
    def __init__(self, obs_dim: int, act_dim: int,
                 act_min: np.ndarray, act_max: np.ndarray, std_min: float, std_max: float,
                 policy_hidden: int, q_hidden: int,
                 actor_lr: float, critic_lr: float,
                 discount_factor: float, update_rate: float,
                 autotune=True, temperature=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GaussianPolicy(obs_dim, act_dim, policy_hidden, torch.tensor(act_min).to(self.device), torch.tensor(act_max).to(self.device), std_min, std_max).to(self.device)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=actor_lr)

        self.critic1 = QNetwork(obs_dim, act_dim, q_hidden).to(self.device)
        self.critic2 = QNetwork(obs_dim, act_dim, q_hidden).to(self.device)
        self.critic1_target = QNetwork(obs_dim, act_dim, q_hidden).to(self.device)
        self.critic2_target = QNetwork(obs_dim, act_dim, q_hidden).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

        if autotune:
            self.target_entropy = -act_dim
            self.log_temperature = torch.zeros(1, requires_grad=True, device=self.device)
            self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=critic_lr)
        else:
            self.temperature = temperature

        self.autotune = autotune

        self.discount_factor = discount_factor
        self.update_rate = update_rate

    def sample_action(self, state, deterministic=False):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
            action, _ = self.actor.get_action(state, deterministic)
            return action.detach().cpu().numpy()
        else:
            action, _ = self.actor.get_action(state, deterministic)
            return action.detach()

    def update(self, states, actions, rewards, next_states, dones,
               update_target=True, update_policy=True, policy_update_iterations=1, update_critic=True):
        if self.autotune:
            temperature = self.log_temperature.exp().item()
        else:
            temperature = self.temperature

        if not torch.is_tensor(states):
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device).squeeze()
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones)).long().to(self.device)

        if update_critic:
            with torch.no_grad():
                next_states_actions, next_states_log_actions = self.actor.get_action(next_states, False, True)
                q1_next_target = self.critic1_target(next_states, next_states_actions)
                q2_next_target = self.critic2_target(next_states, next_states_actions)
                min_critic_next_target = torch.min(q1_next_target, q2_next_target) - temperature * next_states_log_actions
                q_target = rewards + (1 - dones) * self.discount_factor * min_critic_next_target.squeeze()

            q1 = self.critic1(states, actions).squeeze()
            q2 = self.critic2(states, actions).squeeze()
            q1_loss = torch.nn.functional.mse_loss(q1, q_target)
            q2_loss = torch.nn.functional.mse_loss(q2, q_target)
            critic_loss = q1_loss + q2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        if update_policy:
            for _ in range(policy_update_iterations):
                action, log_action = self.actor.get_action(states, False, True)
                q1 = self.critic1(states, action).squeeze()
                q2 = self.critic2(states, action).squeeze()
                q_min = torch.min(q1, q2)
                actor_loss = ((temperature * log_action.squeeze()) - q_min).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_action = self.actor.get_action(states, False, True)
                    temperature_loss = (-self.log_temperature.exp() * (log_action.squeeze() + self.target_entropy)).mean()

                    self.temperature_optimizer.zero_grad()
                    temperature_loss.backward()
                    self.temperature_optimizer.step()

        # update the target networks
        if update_target:
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.update_rate * param.data + (1 - self.update_rate) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.update_rate * param.data + (1 - self.update_rate) * target_param.data)

    def get_value_function(self, states):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_actions, next_states_log_actions = self.actor.get_action(states, False, True)
        q1_next_target = self.critic1_target(states, next_states_actions)
        q2_next_target = self.critic2_target(states, next_states_actions)
        v = torch.min(q1_next_target, q2_next_target) - states * next_states_log_actions

        return v.detach().cpu().numpy()

    def get_torch_value_function(self, states):
        states = states.to(dtype=torch.float32)
        next_states_actions, next_states_log_actions = self.actor.get_action(states, False, True)
        q1_next_target = self.critic1_target(states, next_states_actions)
        q2_next_target = self.critic2_target(states, next_states_actions)
        v = torch.min(q1_next_target, q2_next_target) - states * next_states_log_actions

        return v

