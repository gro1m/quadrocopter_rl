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
        self.target_pos     = target_pos if target_pos is not None \
                              else np.array([0., 0., 10.]) 
    
    def get_reward(self):
        position            = self.sim.pose[:3] #current position
        velocity            = self.sim.v        #current velocity
        dx, dy, dz          = position[0] - self.target_pos[0], \
                              position[1] - self.target_pos[1], \
                              position[2] - self.target_pos[2]
                
        lateral_distance    = (dx + dy)**2
        vertical_distance   = dz**2
        '''
        reward              = -10*(self.sim.v[0]!=0)\
                              -10*(self.sim.v[1]!=0)\
                              +50*(self.sim.v[2] > 0)\
                              -1*self.sim.pose[3:6].sum()\
                              +100/vertical_distance*(vertical_distance != 0)\
                              -lateral_distance
        '''
        reward              = - 0.00034 * vertical_distance**2 + 0.034*velocity[2]
                            
        return reward #np.clip(reward, -1, 1)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.v_previous = self.sim.v #store previous velocity
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #cumulative reward: good? --> action repeats --> needed to learn velocity as only position is incorporated into the state
            #reward  = self.get_reward(reward)
            #penalize crash
            if done and self.sim.time < self.sim.runtime:
                reward         = -1
            
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
           
    #def get_reward(self, reward):
        """Uses current pose of sim to return reward."""
        '''
        #Use distance from point to line
        P, A, B             = self.sim.pose[:3], self.init_pose[:3], self.target_pos
        AP, AB              = A-P, A-B
        n                   = AB/np.linalg.norm(AB)
        pos_dot_norm        = np.linalg.norm(P)*np.linalg.norm(AB) + 0.0001
        distance            = np.linalg.norm(AP-(AP*n)*n)
        
        PB                   = P-B
        '''
        '''
        cPOS, TARG                            = self.sim.pose[:3], self.target_pos
        cPOS_TARG                             = TARG - cPOS
        distances                             = [cPOS_TARG[0], cPOS_TARG[1], cPOS_TARG[2]]

        idx_maxdist, idx_middist, idx_mindist = np.argsort(np.abs(distances))
        
        v_change                              = self.sim.v - self.v_previous
        vx_change, vy_change, vz_change       = v_change[0], v_change[1], v_change[2]
        vxyz_change                           = abs(vx_change)+abs(vy_change)+abs(vz_change)
        
        v_idxmaxdist                          = v_change[idx_maxdist]
        v_idxmiddist                          = v_change[idx_middist]
        v_idxmindist                          = v_change[idx_mindist]
        
        if abs(self.sim.v[0]) > abs(self.sim.v[idx_maxdist]):
            reward                           -= 10
        if abs(self.sim.v[1]) > abs(self.sim.v[idx_maxdist]):
            reward                           -= 10
        if abs(self.sim.v[2]) > abs(self.sim.v[idx_maxdist]):
            reward                           -= 10
        
             
        for i in range(3):
            if self.sim.v[i] == 0 and cPOS_TARG[i] == 0:
                reward                       += 10
                
        if np.sign(cPOS_TARG[idx_maxdist]) == 1 and cPOS_TARG[idx_maxdist] != 0:
            if self.sim.v[idx_maxdist] > 0:
                reward                       += 50
        
        if np.sign(cPOS_TARG[idx_maxdist]) == -1 and cPOS_TARG[idx_maxdist] != 0:
            if self.sim.v[idx_maxdist] < 0:
                reward                       += 50
            
        if np.sign(cPOS_TARG[idx_middist]) == 1 and cPOS_TARG[idx_middist] != 0:
            if self.sim.v[idx_middist] > 0:
                reward                       += 20
        
        if np.sign(cPOS_TARG[idx_middist]) == -1 and cPOS_TARG[idx_middist] != 0:
            if self.sim.v[idx_middist] < 0:
                reward                       += 20
                
        if np.sign(cPOS_TARG[idx_mindist]) == 1 and cPOS_TARG[idx_mindist] != 0:
            if self.sim.v[idx_mindist] > 0:
                reward                       += 10
        
        if np.sign(cPOS_TARG[idx_mindist]) == -1 and cPOS_TARG[idx_mindist] != 0:
            if self.sim.v[idx_mindist] < 0:
                reward                       += 10
            
        reward                               -= np.linalg.norm(distances)
        reward                               += 100/np.linalg.norm(distances)
        reward                               -= 5*min(vxyz_change, 0.1)
        '''
        '''
        reward -= 1.-.3*abs(self.sim.pose[2] - self.target_pos[2])
        reward  = np.clip(reward, -1, 1)
        '''
        '''
        #in order to takeoff, the drone needs to maximize his vertical velocity and minimize lateral velocity.
        if self.sim.v[0] != 0:
            reward           -= 10

        if self.sim.v[1] != 0:
            reward           -= 10

        if self.sim.v[2] > 0:
            reward           += 50
            
        # Then the drone needs to keep being stable and avoid rotation. That's why penalize every increase in euler angles.

        euler_angles          = self.sim.pose[3:6].sum()
        reward               -= euler_angles
        # And to conclude the drone needs to reach the height goal of say 10, that's why reward him when he approach it and  
        # penalize him when he get far away from it.

        current_pos           = self.sim.pose[:3]
        lateral_distance      = ((current_pos[0] - self.target_pos[0]) + (current_pos[1] - self.target_pos[1]))**2
        reward               -= lateral_distance
        vertical_distance     = (current_pos[2] - self.target_pos[2])**2
        if vertical_distance != 0:
            reward           += 100 / vertical_distance
        #(1-5*distance-5*min(velocity_difference, 0.1))#+100/near_to_goal_measure)

        '''
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
        #return np.clip(reward, -1, 1)
        #return reward