import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb

from agents.nn.hyper_q_network import Hyper_Critic
from agents.nn.hyper_policy import Hyper_Policy
from agents.nn.models import Encoder

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, h_dim, param_repeat, latent_state_dim=20, use_encoder=False,
                 param_dim=2, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, hyper_lr=5e-5,
                 policy_lr=1e-6, logger=None, device='cuda'):

        actor = Hyper_Policy

        if use_encoder:
            state_dim = latent_state_dim + param_dim

        task_dim = state_dim
        if param_repeat:
            self.extended_dim = max(1, int(0.5 * (state_dim - param_dim) // param_dim))
            task_dim = (state_dim - param_dim) + param_dim * self.extended_dim
        self.param_repeat = param_repeat

        self.actor = actor(task_dim=task_dim, state_dim=task_dim, action_dim=action_dim, max_action=max_action,
                           h_dim=h_dim).to(device)
        self.actor_target = actor(task_dim=task_dim, state_dim=task_dim, action_dim=action_dim, max_action=max_action,
                                  h_dim=h_dim).to(device)
        self.copy_params(self.actor_target, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)

        critic = Hyper_Critic
        lr = hyper_lr

        self.critic = critic(state_dim=task_dim, action_dim=action_dim+task_dim, h_dim=h_dim).to(device)
        self.critic_target = critic(state_dim=task_dim, action_dim=action_dim+task_dim, h_dim=h_dim).to(device)
        self.copy_params(self.critic_target, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder = Encoder(z_dim=latent_state_dim, h_dim=h_dim).to(device)
            self.critic_optimizer = torch.optim.Adam([{'params': self.critic.parameters()},
                                                      {'params': self.encoder.parameters()}], lr=lr)


        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.logger = logger

        self.param_dim = param_dim

        self.device = device
        self.epoch_count = 0

        self.critic.eval()
        self.actor.eval()
        if self.use_encoder:
            self.encoder.eval()

    def select_action(self, state):
        param = np.copy(state)
        state = state.reshape(1, -1)
        state = state[:, :-self.param_dim]
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if self.use_encoder:
            state = self.encoder(state)
        param = torch.FloatTensor(param.reshape(1, -1)).to(self.device)
        if self.param_repeat:
            p = param[:, -self.param_dim:]
            p = p.repeat([1, self.extended_dim])
            s = param[:, :-self.param_dim]
            if self.use_encoder:
                s = self.encoder(s)
            param = torch.cat([s, p], dim=1)
            state = torch.cat([s, p], dim=1)
        return self.actor(state, param).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, discount=0.99, batch_size=100, debug=False, train_actor=True, noise_clip_flag=False, log=True):
        self.epoch_count += 1

        self.critic.train()
        self.actor.train()
        if self.use_encoder:
            self.encoder.train()

        self.discount = discount

        for it in range(iterations):
            # Sample replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            not_done = 1.0 - done
            param = torch.clone(state)
            state = state[:, :-self.param_dim]
            next_param = torch.clone(next_state)
            next_state = next_state[:, :-self.param_dim]

            if self.use_encoder:
                state = self.encoder(state)
                next_state = self.encoder(next_state)

            if self.param_repeat:
                p = param[:, -self.param_dim:]
                p = p.repeat([1, self.extended_dim])
                s = param[:, :-self.param_dim]
                if self.use_encoder:
                    s = self.encoder(s)
                param = torch.cat([s, p], dim=1)
                state = torch.cat([s, p], dim=1)

                np = next_param[:, -self.param_dim:]
                np = np.repeat([1, self.extended_dim])
                ns = next_param[:, :-self.param_dim]
                if self.use_encoder:
                    ns = self.encoder(ns)
                next_param = torch.cat([ns, np], dim=1)
                next_state = torch.cat([ns, np], dim=1)

            with torch.no_grad():

                if noise_clip_flag:
                    # Select action according to policy and add clipped noise
                    noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    next_action = (self.actor_target(next_state.detach(), next_param) + noise).clamp(-self.max_action, self.max_action)
                else:
                    next_action = self.actor_target(next_state.detach(), next_param)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_param, torch.cat([next_state, next_action], axis=1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(param, torch.cat([state, action], axis=1))
            # self.loss1 = F.mse_loss(current_Q1, target_Q)
            # self.loss2 = F.mse_loss(current_Q2, target_Q)
            self.loss1 = F.huber_loss(current_Q1, target_Q)
            self.loss2 = F.huber_loss(current_Q2, target_Q)

            # Compute critic loss
            critic_loss = self.loss1 + self.loss2

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

        #self.logger['critic'][-1].append(self.compute_gradient(self.critic.q1.parameters()))
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:
                #if train_actor:

                    # Compute actor loss
                    # update to min (Q) for hyper and Q1 for regular

                actor_loss = -self.critic.Q1(param.detach(), torch.cat([state.detach(), self.actor(state.detach(), param.detach())], axis=1), None).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                self.disable_gradients()
                actor_loss.backward()
                #self.logger['policy'][-1].append(self.compute_gradient(self.actor.parameters()))
                self.enable_gradients()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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
        #return self.logger

    def compute_gradient(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
        return total_norm.item()

    def disable_gradients(self):
        for p in self.critic.parameters():
            p.requires_grad = False

    def enable_gradients(self):
        for p in self.critic.parameters():
            p.requires_grad = True

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), directory + filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), directory + filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), directory + filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), directory + filename + "_actor_optimizer")

    def reduce_lr(self, net_optimizer, lr=1e-5):
        for param_group in net_optimizer.param_groups:
            if param_group['lr'] != lr:
                print("### lr drop %s ###" % lr)
            param_group['lr'] = lr

    def load(self, filename, directory):
        # self.critic.load_state_dict(torch.load(directory + filename + "_critic"))
        # self.critic_optimizer.load_state_dict(torch.load(directory + filename + "_critic_optimizer"))
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(directory + filename + "_actor", map_location=self.device))
        #self.actor_optimizer.load_state_dict(torch.load(directory + filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def copy_params(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)