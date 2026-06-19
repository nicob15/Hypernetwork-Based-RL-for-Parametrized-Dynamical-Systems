import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .nn.networks import Actor, Critic, DeepActor, DeepCritic, IQN, DeepIQN
import sys
import numpy as np
import random
import copy
from .nn.intrinsic_curiosity_module import ICM, Inverse, Forward
import os
# setting path
sys.path.append('../utils')
# importing
from utils.utils import ReplayBufferD4PG as ReplayBuffer
from utils.utils import PrioritizedReplay
import wandb

class D4PG(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_dim, action_dim, param_dim, max_action, h_dim, n_step, per, munchausen, distributional,
                 D2RL, curiosity, seed, GAMMA=0.99, TAU=5e-4, LR_ACTOR=1e-4, LR_CRITIC=1e-4, WEIGHT_DECAY=0,
                 LEARN_EVERY=1, LEARN_NUMBER=1, device="cuda"):

        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_dim
        self.action_size = action_dim
        self.param_size = param_dim
        # self.BUFFER_SIZE = BUFFER_SIZE
        #self.BATCH_SIZE = BATCH_SIZE
        self.per = per
        self.munchausen = munchausen
        self.n_step = n_step
        self.distributional = distributional
        self.D2RL = D2RL
        self.curiosity = curiosity[0]
        self.reward_addon = curiosity[1]
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        # self.EPSILON_DECAY = EPSILON_DECAY
        self.device = device
        self.seed = random.seed(seed)
        # distributional Values
        self.N = 64
        self.entropy_coeff = 0.001
        # munchausen values
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9

        self.eta = torch.FloatTensor([.1]).to(device)

        print("Using: ", device)

        # Actor Network (w/ Target Network)
        if not self.D2RL:
            self.actor_local = Actor(state_size=self.state_size, action_size=self.action_size, max_action=max_action,
                                     seed=seed, hidden_size=h_dim).to(device)
            self.actor_target = Actor(state_size=self.state_size, action_size=self.action_size, max_action=max_action,
                                     seed=seed, hidden_size=h_dim).to(device)
        else:
            self.actor_local = DeepActor(state_size=self.state_size, action_size=self.action_size, max_action=max_action,
                                     seed=seed, hidden_size=h_dim).to(device)
            self.actor_target = DeepActor(state_size=self.state_size, action_size=self.action_size, max_action=max_action,
                                     seed=seed, hidden_size=h_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        if self.distributional:
            if not self.D2RL:
                self.critic_local = IQN(self.state_size, self.action_size, layer_size=h_dim, device=device,
                                        seed=seed, dueling=None, N=self.N).to(device)
                self.critic_target = IQN(self.state_size, self.action_size, layer_size=h_dim, device=device,
                                         seed=seed, dueling=None, N=self.N).to(device)
            else:
                self.critic_local = DeepIQN(self.state_size, self.action_size, layer_size=h_dim, device=device,
                                            seed=seed, dueling=None, N=self.N).to(device)
                self.critic_target = DeepIQN(self.state_size, self.action_size, layer_size=h_dim, device=device,
                                             seed=seed, dueling=None, N=self.N).to(device)
        else:
            if not self.D2RL:
                self.critic_local = Critic(self.state_size, self.action_size, seed).to(device)
                self.critic_target = Critic(self.state_size, self.action_size, seed).to(device)
            else:
                self.critic_local = DeepCritic(self.state_size, self.action_size, seed).to(device)
                self.critic_target = DeepCritic(self.state_size, self.action_size, seed).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        print("Actor: \n", self.actor_local)
        print("\nCritic: \n", self.critic_local)

        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size)
            forward_m = Forward(self.state_size, self.action_size, inverse_m.calc_input_layer(), device=device)
            self.icm = ICM(inverse_m, forward_m, device=device)  # .to(device)
            print(inverse_m, forward_m)

        # Noise process
        #self.epsilon = 0.3
        # self.noise_type = noise_type
        # if noise_type == "ou":
        #     self.noise = OUNoise(action_size, random_seed)
        #     self.epsilon = EPSILON
        # else:
        #     self.epsilon = 0.3
        # print("Use Noise: ", noise_type)
        # Replay memory
        # if per:
        #     self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, device=device, seed=random_seed, gamma=GAMMA,
        #                                     n_step=n_step, parallel_env=worker, beta_frames=frames)
        #
        # else:
        #     self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_step=n_step, parallel_env=worker, device=device,
        #                                seed=random_seed, gamma=GAMMA)

        if distributional:
            self.train = self.learn_distribution
        else:
            self.train = self.learn_

        print("Using PER: ", per)
        print("Using Munchausen RL: ", munchausen)

    # def step(self, state, action, reward, next_state, done, timestamp, writer):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     # Save experience / reward
    #     self.memory.add(state, action, reward, next_state, done)
    #
    #     # Learn, if enough samples are available in memory
    #     if len(self.memory) > self.BATCH_SIZE and timestamp % self.LEARN_EVERY == 0:
    #         for _ in range(self.LEARN_NUMBER):
    #             experiences = self.memory.sample()
    #
    #             losses = self.train(experiences, self.GAMMA)
            # writer.add_scalar("Critic_loss", losses[0], timestamp)
            # writer.add_scalar("Actor_loss", losses[1], timestamp)
            # if self.curiosity:
            #     writer.add_scalar("ICM_loss", losses[2], timestamp)

    def select_action(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).reshape(1,-1).float().to(self.device)

        assert state.shape == (state.shape[0], self.state_size), "shape: {}".format(state.shape)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # if add_noise:
        #     if self.noise_type == "ou":
        #         action += self.noise.sample() * self.epsilon
        #     else:
        #         action += self.epsilon * np.random.normal(0, scale=1)
        return action.flatten()  # np.clip(action, -1, 1)

    # def reset(self):
    #     self.noise.reset()

    def learn_(self, memory, discount, batch_size, iterations, log=False):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        for _ in range(iterations):
            experiences = memory.sample(batch_size=batch_size)

            states, actions, rewards, next_states, dones, idx, weights = experiences
            icm_loss = 0
            # calculate curiosity
            if self.curiosity:
                icm_loss, forward_pred_err = self.icm.calc_errors(state1=states, state2=next_states, action=actions)
                r_i = self.eta * forward_pred_err
                assert r_i.shape == rewards.shape, "r_ and r_e have not the same shape"

                if self.reward_addon == 1:
                    rewards += r_i.detach()
                else:
                    rewards = r_i.detach()

            # ---------------------------- update critic ---------------------------- #
            if not self.munchausen:
                # Get predicted next-state actions and Q values from target models
                with torch.no_grad():
                    actions_next = self.actor_target(next_states.to(self.device))
                    Q_targets_next = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
                    # Compute Q targets for current states (y_i)
                    Q_targets = rewards + (discount ** self.n_step * Q_targets_next * (1 - dones))
            else:
                with torch.no_grad():
                    actions_next = self.actor_target(next_states.to(self.device))
                    q_t_n = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
                    # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                    # this might cause some instability (?) needs to be tested
                    logsum = torch.logsumexp(q_t_n / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
                    assert logsum.shape == (batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                    tau_log_pi_next = (q_t_n - self.entropy_tau * logsum)

                    pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1)
                    # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                    Q_target = (self.GAMMA ** self.n_step * (pi_target * (q_t_n - tau_log_pi_next) * (1 - dones)))
                    assert Q_target.shape == (batch_size, 1), "has shape: {}".format(Q_target.shape)

                    q_k_target = self.critic_target(states, actions)
                    tau_log_pik = q_k_target - self.entropy_tau * torch.logsumexp(q_k_target / self.entropy_tau, 1).unsqueeze(-1)
                    assert tau_log_pik.shape == (batch_size, 1), "shape instead is {}".format(tau_log_pik.shape)
                    # calc munchausen reward:
                    munchausen_reward = (rewards + self.alpha * torch.clamp(tau_log_pik, min=self.lo, max=0))
                    assert munchausen_reward.shape == (batch_size, 1)
                    # Compute Q targets for current states
                    Q_targets = munchausen_reward + Q_target
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            if self.per:
                td_error = Q_targets - Q_expected
                critic_loss = (td_error.pow(2) * weights.to(self.device)).mean().to(self.device)
            else:
                critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)
            if self.per:
                memory.update_priorities(idx, np.clip(abs(td_error.data.cpu().numpy()), -1, 1))
            # ----------------------- update epsilon and noise ----------------------- #

            #self.epsilon *= self.EPSILON_DECAY

            # log training losses
            if log:
                wandb.log({'train/critic_loss': critic_loss.detach().cpu().numpy(),
                        'train/actor_loss': actor_loss.detach().cpu().numpy(),
                        'train/icm_loss': icm_loss})


            #if self.noise_type == "ou": self.noise.reset()
        # return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy(), icm_loss

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
    def _to_device_and_flat(self, x):
        """Helper: sposta tensor su device e appiattisce dimensioni extra -> (batch, features)."""
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
        return x

    def learn_distribution(self, memory, discount, batch_size, iterations, log=False):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        for _ in range(iterations):
            experiences = memory.sample(batch_size=batch_size)

            states, actions, rewards, next_states, dones, idx, weights = experiences

            # usa il metodo di classe invece di definire una funzione locale
            states = self._to_device_and_flat(states)
            actions = self._to_device_and_flat(actions)
            next_states = self._to_device_and_flat(next_states)
            if isinstance(dones, torch.Tensor):
                dones = dones.to(self.device)
            else:
                dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.to(self.device)
            else:
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models

            # Get max predicted Q values (for next states) from target model
            if not self.munchausen:
                with torch.no_grad():
                    next_actions = self.actor_local(next_states)
                    Q_targets_next, _ = self.critic_target(next_states, next_actions, self.N)
                    Q_targets_next = Q_targets_next.transpose(1, 2)
                # Compute Q targets for current states
                Q_targets = rewards.unsqueeze(-1) + (discount ** self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
            else:
                with torch.no_grad():
                    #### CHECK FOR THE SHAPES!!
                    actions_next = self.actor_target(next_states.to(self.device))
                    Q_targets_next, _ = self.critic_target(next_states.to(self.device), actions_next.to(self.device), self.N)

                    q_t_n = Q_targets_next.mean(1)
                    # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                    # this might cause some instability (?) needs to be tested
                    logsum = torch.logsumexp(q_t_n / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
                    assert logsum.shape == (batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                    tau_log_pi_next = (q_t_n - self.entropy_tau * logsum).unsqueeze(1)

                    pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)
                    # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                    Q_target = (discount ** self.n_step * (pi_target * (Q_targets_next - tau_log_pi_next) * (1 - dones.unsqueeze(-1)))).transpose(1, 2)
                    assert Q_target.shape == (batch_size, self.action_size, self.N), "has shape: {}".format(Q_target.shape)

                    q_k_target = self.critic_target.get_qvalues(states, actions)
                    tau_log_pik = q_k_target - self.entropy_tau * torch.logsumexp(q_k_target / self.entropy_tau, 1).unsqueeze(-1)
                    assert tau_log_pik.shape == (batch_size, self.action_size), "shape instead is {}".format(
                        tau_log_pik.shape)
                    # calc munchausen reward:
                    munchausen_reward = (rewards + self.alpha * torch.clamp(tau_log_pik, min=self.lo, max=0)).unsqueeze(-1)
                    assert munchausen_reward.shape == (batch_size, self.action_size, 1)
                    # Compute Q targets for current states
                    Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            Q_expected, taus = self.critic_local(states, actions, self.N)
            assert Q_targets.shape == (batch_size, 1, self.N)
            assert Q_expected.shape == (batch_size, self.N, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (batch_size, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            if self.per:
                critic_loss = (quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights.to(self.device)).mean()
            else:
                critic_loss = quantil_l.sum(dim=1).mean(dim=1).mean()
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local.get_qvalues(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)
            if self.per:
                memory.update_priorities(idx, np.clip(abs(td_error.sum(dim=1).mean(dim=1, keepdim=True).data.cpu().numpy()), -1, 1))
            # ----------------------- update epsilon and noise ----------------------- #

            #self.epsilon *= self.EPSILON_DECAY

            # log training losses
            if log:
                wandb.log({'train/critic_loss': critic_loss.detach().cpu().numpy(),
                        'train/actor_loss': actor_loss.detach().cpu().numpy()})

            #if self.noise_type == "ou": self.noise.reset()
        #return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def save(self, filename, directory):
     """Salva i pesi dell'attore e del critico (local + target)"""
     os.makedirs(directory, exist_ok=True)

     torch.save(self.actor_local.state_dict(),  f"{directory}/{filename}_actor_local.pth")
     torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
     torch.save(self.critic_local.state_dict(), f"{directory}/{filename}_critic_local.pth")
     torch.save(self.critic_target.state_dict(),f"{directory}/{filename}_critic_target.pth")

     print("✔ D4PG weights saved.")


    def load(self, filename, directory):
     """Carica i pesi di actor e critic (local + target)"""
     actor_local_path  = f"{directory}/{filename}_actor_local.pth"
     actor_target_path = f"{directory}/{filename}_actor_target.pth"
     critic_local_path = f"{directory}/{filename}_critic_local.pth"
     critic_target_path= f"{directory}/{filename}_critic_target.pth"

     device = self.device

     if os.path.exists(actor_local_path):
        self.actor_local.load_state_dict(torch.load(actor_local_path, map_location=device))
     if os.path.exists(actor_target_path):
        self.actor_target.load_state_dict(torch.load(actor_target_path, map_location=device))
     if os.path.exists(critic_local_path):
        self.critic_local.load_state_dict(torch.load(critic_local_path, map_location=device))
     if os.path.exists(critic_target_path):
        self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))

     print("✔ D4PG weights loaded successfully.")



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


def calc_fraction_loss(FZ_, FZ, taus, weights=None):
    """calculate the loss for the fraction proposal network """
    # usa dimensione derivata da taus invece di valori hard-coded
    N = taus.shape[1]  # es. 32
    assert FZ.shape[1] == N and FZ_.shape[1] == N, "FZ dimension mismatch with taus"

    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:]
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2))
    # gradients expected shape (batch, N-1) -> sommiamo usando taus intermedi
    gradients = gradients.view(taus.shape[0], N-1)
    assert not gradients.requires_grad
    if weights is not None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1) * weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    td_errors Tensor shape: (batch, N, N) expected
    """
    # calcola N dinamicamente
    assert td_errors.ndim == 3, "td_errors must be 3D (batch, N, N)"
    N = td_errors.shape[1]
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], N, N), f"huber loss has wrong shape {loss.shape} expected (batch,{N},{N})"
    return loss

