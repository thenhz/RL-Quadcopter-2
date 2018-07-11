import numpy as np
from agents import Actor, Critic
import pandas as pd
from misc import *
import os


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Episode vars
        self.episode = 0
        self.episode_duration = 0
        self.total_reward = None
        self.best_total_reward = -np.inf
        self.score = None
        self.best_score = -np.inf
        self.last_states = None
        self.last_action = None

        # Actor (Policy) Model
        self.actor_local = Actor.Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor.Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic.Critic(self.state_size, self.action_size)
        self.critic_target = Critic.Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.3
        self.exploration_sigma = 0.4
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Save params
        self.stats_filename = os.path.join(
            './out',
            "stats_{}_{}.csv".format("DDPG", get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters

    def reset_episode(self):

        self.score = self.total_reward / float(self.episode_duration) if self.episode_duration else -np.inf
        if self.best_score < self.score:
            self.best_score = self.score
        if self.total_reward and self.total_reward > self.best_total_reward:
            self.best_total_reward = self.total_reward
        self.total_reward = None
        self.episode_duration = 0
        self.last_states = None
        self.last_action = None
        state = self.task.reset()
        self.episode += 1
        return state

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))  # write header first time only

    def step(self, action, reward, next_state, done):

        #states = self.preprocess_state(states)
        if self.total_reward:
            self.total_reward += reward
        else:
            self.total_reward = reward

        self.episode_duration += 1
        # Save experience / reward
        if self.last_states is not None and self.last_action is not None:
            self.memory.add(self.last_states, self.last_action, reward, next_state, done)

        self.last_states = next_state

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        if done:
            self.write_stats([self.episode, self.total_reward])
#            if self.save_weights_every and self.episode % self.save_weights_every == 0:
#                self.save_weights()

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,
                                                                                                        self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)