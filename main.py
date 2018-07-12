import sys

import numpy as np
from agents.agent import DDPG
from task import Task
import csv
import matplotlib.pyplot as plt

labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward']
results = {x : [] for x in labels}
file_output = 'data.txt'

num_episodes = 50
target_pos = np.array([0., 0., 10.])

init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([600., 600., 600.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities

task = Task(init_pose=init_pose, init_velocities=init_velocities, init_angle_velocities=init_angle_velocities, target_pos=target_pos)
agent = DDPG(task)

with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            if i_episode == num_episodes:
                to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action) + [reward]
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)

            if done:
                #print("\rEpisode = {:4d},  total_reward = {:7.3f} (best = {:7.3f}), Z_FINISH = {:4f}, score = {:7.3f} (best = {:7.3f}), duration= {:4d}".format(
                #    i_episode, agent.total_reward, agent.best_total_reward, task.sim.pose[2], agent.score, agent.best_score, agent.episode_duration), end="")  # [debug]
                print("\rEpisode = {:4d}", i_episode)
                break
        sys.stdout.flush()

    #plot some stuff
    # reward
    plt.plot(results['time'], results['reward'], label='reward')
    plt.legend()
    _ = plt.ylim()
    plt.show()

    #positions
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    _ = plt.ylim()
    plt.show()

    #velocities
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.legend()
    _ = plt.ylim()
    plt.show()