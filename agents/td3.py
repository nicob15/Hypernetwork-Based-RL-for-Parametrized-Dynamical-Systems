import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from agents.nn.models import Encoder

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, h_dim=10, param_dim=1, param_repeat=False):
        super(Actor, self).__init__()

        if param_repeat:
            self.extended_dim = max(1, int(0.5 * (state_dim - param_dim) // param_dim))
            state_dim = (state_dim - param_dim) + param_dim * self.extended_dim

        self.param_repeat = param_repeat

        self.l1 = nn.Linear(state_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, action_dim)

        torch.nn.init.normal_(self.l3.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(self.l3.bias)


        self.max_action = max_action

    def forward(self, x):
        if self.param_repeat:
            state = x[:, :-1]
            param = x[:, -1:]
            x = torch.cat([state, param.repeat([1, self.extended_dim])], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=10, param_dim=1, param_repeat=False):
        super(Critic, self).__init__()

        if param_repeat:
            self.extended_dim = max(1, int(0.5 * (state_dim - param_dim) // param_dim))
            state_dim = (state_dim - param_dim) + param_dim * self.extended_dim

        self.param_repeat = param_repeat

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, 1)

        torch.nn.init.normal_(self.l3.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(self.l3.bias)


        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, h_dim)
        self.l5 = nn.Linear(h_dim, h_dim)
        self.l6 = nn.Linear(h_dim, 1)

        torch.nn.init.normal_(self.l6.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(self.l6.bias)



    def forward(self, x, u):
        if self.param_repeat:
            state = x[:, :-1]
            param = x[:, -1:]
            x = torch.cat([state, param.repeat([1, self.extended_dim])], dim=1)

        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    def Q1(self, x, u):
        if self.param_repeat:
            state = x[:, :-1]
            param = x[:, -1:]
            x = torch.cat([state, param.repeat([1, self.extended_dim])], dim=1)

        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, h_dim, tau, param_repeat, latent_state_dim=20, param_dim=1,
                 use_encoder=False, device='cuda'):

        if use_encoder:
            state_dim = latent_state_dim + param_dim

        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, h_dim=h_dim, param_repeat=param_repeat).to(device)
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action, h_dim=h_dim, param_repeat=param_repeat).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, h_dim=h_dim, param_repeat=param_repeat).to(device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, h_dim=h_dim, param_repeat=param_repeat).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.tau = tau
        self.device = device

        self.epoch_count = 0

        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder = Encoder(z_dim=latent_state_dim, h_dim=h_dim).to(device)
            self.critic_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},
                                                      {'params': self.encoder.parameters()}], lr=3e-4)

        self.critic.eval()
        self.actor.eval()
        if self.use_encoder:
            self.encoder.eval()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        if self.use_encoder:
            p = state[:, -1].reshape(-1, 1)
            state = self.encoder(state[:, :-1])
            state = torch.cat([state, p], dim=1)

        return self.actor(state).cpu().data.numpy().flatten()

    # regular TD3 is stateless; add this to conform to API
    def reset(self):
        pass

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2, noise_clip_flag=False, log=True):
        self.epoch_count += 1

        self.critic.train()
        self.actor.train()
        if self.use_encoder:
            self.encoder.train()

        for it in range(iterations):

            # Sample replay buffer
            x, u, y, r, d = replay_buffer.sample(batch_size)
            state = x
            action = u
            next_state = y
            reward = r
            done = 1.0 - d

            if self.use_encoder:
                p = state[:, -1].reshape(-1, 1)
                state = self.encoder(state[:, :-1])
                state = torch.cat([state, p], dim=1)

                np = next_state[:, -1].reshape(-1, 1)
                next_state = self.encoder(next_state[:, :-1])
                next_state = torch.cat([next_state, np], dim=1)

            if noise_clip_flag:
                noise = torch.FloatTensor(action.size()).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state.detach()) + noise).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor_target(next_state.detach())

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()
            # import ipdb; ipdb.set_trace()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state.detach(), self.actor(state.detach())).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # log training losses
                if log:
                    if self.epoch_count % 1 == 0:
                        wandb.log({'train/critic_loss': critic_loss,
                                   'train/actor_loss': actor_loss})

        self.critic.eval()
        self.actor.eval()
        if self.use_encoder:
            self.encoder.eval()


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self, '%s/%s_all.pth' % (directory, filename))


    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location='cpu'))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu'))
        else:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

    def load_all(self, filename, directory):
        if not torch.cuda.is_available():
            return torch.load('%s/%s_all.pth' % (directory, filename), map_location='cpu')
        else:
            return torch.load('%s/%s_all.pth' % (directory, filename))