import keras
from .actor_critic import Actor, Critic
import numpy as np
import copy
import random
from collections import deque, namedtuple

class DDPG_Agent():
    "Reinforcement Learning with Deep Deterministic Policy Gradients"
    def __init__(self, task):
        self.task                         = task
        self.state_size, self.action_size = task.state_size, task.action_size
        self.action_low, self.action_high = task.action_low, task.action_high
        
        self.params()
        
        # ACTOR (POLICY) MODEL
        self.actor_local       = Actor(self.state_size, self.action_size,
                                       self.action_low, self.action_high)
        self.actor_target      = Actor(self.state_size, self.action_size,
                                       self.action_low, self.action_high)
        
        # CRITIC (VALUE) MODEL
        self.critic_local      = Critic(self.state_size, self.action_size)
        self.critic_target     = Critic(self.state_size, self.action_size)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        self.noise             = OUNoise(self.action_size, self.exploration_mu,
                                         self.exploration_theta, self.exploration_sigma)
        
        self.memory            = ReplayBuffer(self.buffer_size, self.batch_size)
        
        self.avg_rewards       = 0
        
    def params(self):
        # NOISE PROCESS
        self.exploration_mu    = 0
        self.exploration_theta = 0.3#0.15
        self.exploration_sigma = 0.6#0.20
        
        # REPLAY BUFFER
        self.buffer_size       = 100000 # buffer size
        self.batch_size        = 64
        
        # ALGORITHM PARAMETERS
        self.gamma             = 0.99#0.99   # discount factor
        self.tau               = 0.001   # for soft update of target parameters      
        
    def reset_episode(self):
        self.noise.reset()
        state                  = self.task.reset()
        self.last_state        = state
        return state
    
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size: #learn if enough samples available in memory
            experiences  = self.memory.sample()
            self.learn(experiences) #update weights of value function approximation
        self.last_state  = next_state
        
    def act(self, state):
        "Returns actions for given state as per current policy"
        state            = np.reshape(state, [-1, self.state_size])
        action           = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample()) #add some noise for exploration
    
    def learn(self, experiences):
        "Update policy and value parameters using given batch of experience tuples"
        states           = np.vstack([e.state for e in experiences if e is not None])\
                           .astype(np.float32)
        actions          = np.array([e.action for e in experiences if e is not None])\
                           .astype(np.float32).reshape(-1, self.action_size)
        rewards          = np.array([e.reward for e in experiences if e is not None])\
                           .astype(np.float32).reshape(-1,1)
        next_states      = np.vstack([e.next_state for e in experiences if e is not None])
        dones            = np.vstack([e.done for e in experiences if e is not None])\
                           .astype(np.uint8).reshape(-1,1)
        
        # get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next     = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next   = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        # Compute Q targets for current states and train critic model (local)
        Q_targets        = rewards + self.gamma * Q_targets_next * (1-dones)

        self.critic_local.model.train_on_batch(x = [states, actions], y = Q_targets)
        
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), 
                                      (-1, self.action_size))

        self.actor_local.train_fn([states, action_gradients, 1]) #custom training function, 1 for K.learning_phasse

            
        # Soft-update training models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model,  self.actor_target.model)
        
        self.avg_rewards = np.sum(rewards)/len(rewards)
        
    def soft_update(self, local_model, target_model):
        "Soft update model parameters."
        ''' After training batch of experiences we could just copy newly learned weights from the local model 
          to the target model. Individual experience batches can introduce a lot of variance in the process, 
          so it is better to perform a soft update controlled by parameter tau. '''
        local_weights    = np.array(local_model.get_weights())
        target_weights   = np.array(target_model.get_weights())
        
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        
        new_weights    = self.tau * local_weights + (1-self.tau)*target_weights
        target_model.set_weights(new_weights)
        
class ReplayBuffer:
    "Fixed-size buffer to store experience tuples"
    
    def __init__(self, buffer_size, batch_size):
        "buffer size: max buffer size"
        "batch size: size of each training batch"
        self.memory     = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names = ["state","action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        "Add new experience to memory"
        e               = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
   
    def sample(self, batch_size = 64):
        "Randomly sample experiences batch from memory"
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        "Return current memory size"
        return len(self.memory)
    
class OUNoise:
    "Ornstein-Uhlenbeck process"
    
    def __init__(self, size, mu, theta, sigma):
        "Initialize parameters and noise process"
        self.mu        = mu * np.ones(size)
        self.theta     = theta
        self.sigma     = sigma
        self.reset()
        
    def reset(self):
        "Reset internal state (noise) to mean (mu)"
        self.state     = copy.copy(self.mu)
        
    def sample(self):
        "Update internal state and return it as a noise sample"
        x              = self.state
        dx             = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state     = x + dx
        return self.state 
    
    