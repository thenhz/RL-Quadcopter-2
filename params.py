from keras import layers, models, optimizers
from keras import backend as K
from agents.ou_noise import OUNoise
from agents.replay_buffer import ReplayBuffer
from agents.PrioritizedMemory import  Memory


class actor_params():

    def __init__(self):
        self.lr = 0.001
        self.clipvalue = 0.5

        self.optimizer = optimizers.Adadelta(decay=0.01)

    def build_nn(self, input):
        net = layers.Dense(units=32, activation='relu')(input)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)
        return net

class critic_params():

    def __init__(self):
        self.lr = 0.001
        self.clipvalue = 0.5

        self.optimizer = optimizers.Adadelta(decay=0.01)

    def build_nn(self, actions, states):
        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        # Add more layers to the combined network if needed
        return net


class agent_params():

    def __init__(self, action_size):
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15  # same direction
        self.exploration_sigma = 0.001  # random noise

        self.noise = OUNoise(action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        #self.memory = Memory(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.1  # for soft update of target parameters
