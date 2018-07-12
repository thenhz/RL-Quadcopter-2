import gym
import sys
from agents.agent import DDPG
import numpy as np
from MountainCar_task import MountainCarTask
import matplotlib.pyplot as plt

num_episodes = 1000
task = MountainCarTask()
agent = DDPG(task)

display_graph = True
display_freq = 50
rewards = []
total_rewards = []

for i_episode in range(1, num_episodes + 1):
    state = agent.reset_episode()  # start a new episode

    # prior to the start of each episode, clear the datapoints
    # x, y1, y2 = [], [], []

    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        rewards.append(reward)
        agent.step(action, reward, next_state, done)
        state = next_state
        # within the episode loop
        if (i_episode % display_freq == 0) and (display_graph == True):
            #    x.append(task.sim.time) # time
            #    y1.append(reward) # y-axis 1 values
            #    y2.append(task.sim.pose[2]) # y-axis 2 values

            print(f'Episode number {i_episode}')
            print(
                f'action {action}, reward {reward}, next_state {next_state}, done {done}')
        #    print(f'Plot values - time {task.sim.time}, reward {reward}, z {task.sim.pose[2]}')
        # if done:
        #    print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
        #        i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
        #    break
        if done:
            print("\rEpisode = {:4d}, reward = {:7.3f}".format(
                i_episode, reward), end="")  # [debug]
        #    if (episode % display_freq == 0) and (display_graph == True):
        #               plt_dynamic(x, y1, y2)
        break

    total_rewards.append(np.average(rewards))

    sys.stdout.flush()

plt.plot(total_rewards)
plt.legend()
_ = plt.ylim()
plt.show()
