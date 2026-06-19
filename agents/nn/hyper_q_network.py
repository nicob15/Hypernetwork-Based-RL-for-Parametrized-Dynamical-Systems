from agents.nn.hyper_network import Hyper_QNetwork
from agents.nn.hyper_network import Hyper_QNetwork_3layers
import torch.nn as nn

class Hyper_Critic(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self, state_dim, action_dim, h_dim):
        super(Hyper_Critic, self).__init__()
        meta_v_dim = state_dim
        base_v_dim = action_dim
        self.q1 = Hyper_QNetwork(meta_v_dim, base_v_dim, h_dim)
        self.q2 = Hyper_QNetwork(meta_v_dim, base_v_dim, h_dim)

    def forward(self, state, action, debug=None):
        q1 = self.q1(state, action, debug)
        q2 = self.q2(state, action)
        return q1, q2

    def Q1(self, state, action, debug):
        q1 = self.q1(state, action, debug)
        return q1

# class Hyper_Critic_3layers(nn.Module):
#     # Hyper net that create weights from the state for a net that estimates function Q(S, A)
#     def __init__(self, state_dim, action_dim, h_dim):
#         super(Hyper_Critic_3layers, self).__init__()
#         meta_v_dim = state_dim
#         base_v_dim = action_dim
#         self.q1 = Hyper_QNetwork_3layers(meta_v_dim, base_v_dim, h_dim)
#         self.q2 = Hyper_QNetwork_3layers(meta_v_dim, base_v_dim, h_dim)
#
#     def forward(self, state, action, debug=None):
#         q1 = self.q1(state, action, debug)
#         q2 = self.q2(state, action)
#         return q1, q2
#
#     def Q1(self, state, action, debug):
#         q1 = self.q1(state, action, debug)
#         return q1