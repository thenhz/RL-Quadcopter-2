import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 6

        self.state_size = self.action_repeat * 6
        self.action_low = 800  # constrain propeller speeds to reasonable range
        self.action_high = 900
        self.action_size = 4

        self.total_reward = 0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        x = self.sim.pose[0]
        y = self.sim.pose[1]
        z = self.sim.pose[2]

        vz = self.sim.v[2]

        phi = self.sim.pose[3]
        theta = self.sim.pose[4]
        psi = self.sim.pose[5]

        reward = -abs(self.target_pos[2] - z)
        return reward

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            rotor_speeds = rotor_speed * 4 # constrain all rotors to same speed
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        self.total_reward += reward
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.total_reward = 0
        return state
