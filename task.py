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
        self.sim            = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat  = 3

        self.state_size     = self.action_repeat * 6
        self.action_low     = 0
        self.action_high    = 900
        self.action_size    = 4
        
        self.init_pose      = init_pose
        
        self.v_previous     = [0., 0., 0.]

        # Goal
        self.target_pos     = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #Use distance from point to line
        P, A, B             = self.sim.pose[:3], self.init_pose[:3], self.target_pos
        AP, AB              = A-P, A-B
        n                   = AB/np.linalg.norm(AB)
        pos_dot_norm        = np.linalg.norm(P)*np.linalg.norm(AB) + 0.0001
        distance            = np.linalg.norm(AP-(AP*n)*n)
        
        PB                   = P-B

        velocity_difference  = abs((self.sim.v - self.v_previous)[0])+\
                               abs((self.sim.v - self.v_previous)[1])+\
                               abs((self.sim.v - self.v_previous)[2])
        reward               = (1-5*distance-5*min(velocity_difference, 0.1))#+100/near_to_goal_measure)
        ''' 
        OTHER IDEAS:                     
        vth                = 8
        vel_measure        = (abs(self.sim.v[0])-vth)/vth + (abs(self.sim.v[1])-vth)/vth +\
                             (abs(self.sim.v[2])-vth)/vth
        near_to_goal_measure = max([abs(PB[0]), abs(PB[1]), abs(PB[2])])+0.0001
        clip gradients to avoid instabilites with np.tanh (seems not to be useful)
        reward function tries: 
        (1) +5*near_to_goal - vel_measure
        (2) np.tanh(- 1000*pos_measure - vel_measure)
        (3) original proposal: 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        (4) np.dot(self.sim.pose[:3],self.target_pos)/norm#np.dot(self.sim.pose[:3],self.target_pos)/norm
        '''
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.v_previous = self.sim.v #store previous velocity
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            #penalize crash
            '''
            if done and self.sim.time < self.sim.runtime:
                reward         = -1
            '''
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state