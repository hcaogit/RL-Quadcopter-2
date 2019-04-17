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

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        self.prev_pos = init_pose

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.0 - 0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = self.product_reward()
        #reward = self.distance_reward()
        
        rewardRange = 1.0
        if reward > rewardRange:
            reward = rewardRange
        elif reward < -rewardRange:
            reward = -rewardRange
        return reward

    def distance_reward(self):
        init_to_target = self.target_pos - self.sim.init_pose[:3]
        cur_pos = self.sim.pose[:3]
        cur_to_target = self.target_pos - cur_pos
        prev_to_target = self.target_pos - self.prev_pos
        
        if np.linalg.norm(init_to_target) == 0 : 
            if np.linalg.norm(cur_to_target) == 0 :
                return 0.1
            else :
                return -0.1
        else :
            if np.linalg.norm(cur_to_target) < np.linalg.norm(prev_to_target) :
                return 0.1 
            else :
                return -0.1
        
    def product_reward(self):
        cur_pos = self.sim.pose[:3]
        to_target = self.target_pos - cur_pos
        to_cur = cur_pos-self.prev_pos
        
        to_target_norm = np.linalg.norm(to_target)
        to_cur_norm = np.linalg.norm(to_cur)
        
        reward = 0
        delta = 0.1
        if to_target_norm == 0 :
            reward = delta
            if to_cur_norm == 0 : 
                reward = reward + delta
        elif to_cur_norm == 0 :
            reward = -delta
        
        if reward != 0 :
            return reward
        
        #normal = to_target_norm * to_cur_norm
        dot = to_target.dot(to_cur)
        cross = np.linalg.norm(np.cross(to_target, to_cur))
        reward = dot*0.01 - cross*0.01
        
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.prev_pos = self.sim.pose[:3]
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state