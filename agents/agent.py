from agents.actor import Actor
from agents.critic import Critic
import numpy as np

from params import *

class Agent():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        actor_local_params = actor_params()
        actor_target_params = actor_params()
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 params=actor_local_params)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                  params=actor_target_params)

        # Critic (Value) Model
        critic_local_params = critic_params()
        critic_target_params = critic_params()
        self.critic_local = Critic(self.state_size, self.action_size, params=critic_local_params)
        self.critic_target = Critic(self.state_size, self.action_size, params=critic_target_params)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        agent_par = agent_params(self.action_size)
        # Noise process
        self.noise = agent_par.noise
        # Replay memory
        self.batch_size = agent_par.batch_size
        self.memory = agent_par.memory
        # Algorithm parameters
        self.gamma = agent_par.gamma # discount factor
        self.tau = agent_par.tau # for soft update of target parameters

        
        # Compute the ongoing top score
        self.top_score = -np.inf
        self.score = 0

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            b_idx, experiences, b_ISWeights = self.memory.sample()
            self.learn(experiences, b_idx, b_ISWeights)

        # Roll over last state and action
        self.last_state = next_state

        # stats
        self.score += reward
        if done:
            if self.score > self.top_score:
                self.top_score = self.score

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences, b_idx=None, b_ISWeights=None):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
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
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        if b_idx is not None:
            self.memory.batch_update(b_idx, np.abs(Q_targets_next - Q_targets))


    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)