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
        self.action_repeat = 3

        # State
        self.state_size = self.action_repeat * (9)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 150.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        penalty = 0
        current_position = self.sim.pose[:3]
        
        # PENALTIES
        
        #penalty for distance from target on x axis
        penalty += abs(current_position[0]-self.target_pos[0])**2
        #penaltyfor distance from target on y axis
        penalty += abs(current_position[1]-self.target_pos[1])**2
        #penaltyfor distance from target on z axis, weighted as this is more important for this task of hovering at a certain height
        penalty += 12 * abs(current_position[2]-self.target_pos[2])**2
        #penalty for uneven takeoff
        penalty += abs(self.sim.pose[3:6]).sum()
        #penalty for being far away from target and travelling fast
        penalty += 50* abs(abs(current_position-self.target_pos).sum() - abs(self.sim.v).sum())

        # REWARDS
        
        #ongoing reward for being airbourne
        if current_position[2] > 0.0:
            reward += 100
        #additional reward for flying near the target, where each x,y,z axis needs to be itself close to the target point for the agent to be rewarded
        if np.sqrt((current_position[0]-self.target_pos[0])**2) < 10 and np.sqrt((current_position[1]-self.target_pos[1])**2) < 10 and np.sqrt((current_position[2]-self.target_pos[2])**2) < 10:
            reward += 1000

        # TOTAL
        
        return reward - (penalty * 0.0002)

    def cur_state(self):
        state = np.concatenate([np.array(self.sim.pose), np.array(self.sim.v)])
        return state    
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            state = self.cur_state()
            pose_all.append(self.cur_state())
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.cur_state()] * self.action_repeat) 
        return state
