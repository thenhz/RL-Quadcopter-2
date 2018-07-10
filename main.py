import sys

import numpy as np
from agents.agent import DDPG
from task import Task

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = DDPG(task)

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d},  total_reward = {:7.3f} (best = {:7.3f}), Z_FINISH = {:4f}, score = {:7.3f} (best = {:7.3f}), duration= {:4d}".format(
                i_episode, agent.total_reward, agent.best_total_reward, task.sim.pose[2], agent.score, agent.best_score, agent.episode_duration), end="")  # [debug]
            break
    sys.stdout.flush()