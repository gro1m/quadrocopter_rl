# Put Training here
from agents.agent import *
from task import Task
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

def train(num_episodes = 1000, init_pos = np.array([0., 0., 1.0, 0.0, 0.0, 0.0]), 
          target_pos = np.array([0., 0., 10.]), do_plot = 'yes'):
    task         = Task(init_pose = init_pos, target_pos=target_pos)
    agent        = DDPG_Agent(task)
    coords       = []
    rewards      = []
    for i_episode in range(1, num_episodes+1):
        state    = agent.reset_episode() # start a new episode
        while True:
            # ACT BASED ON CURRENT STATE
            action = agent.act(state) 
            # DO PHYSICS SIMULATION ("FLY")
            x,y,z = task.sim.pose[:3]
            next_state, reward, done = task.step(action)
            # UPDATE EXPERIENCE AND LEARN IF POSSIBLE
            agent.step(action, reward, next_state, done)
            # ASSIGN next_state TO state
            state = next_state
            if done:
                print("\rEpisode = {:4d}, average rewards = {:7.3f}"\
                      .format(i_episode, agent.avg_rewards), end="")  # [debug]
                rewards.append(reward/agent.task.action_repeat)
                coords.append([x,y,z])
                break
            sys.stdout.flush()

    if do_plot == 'yes':
        fig = plt.figure(1)
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        # Data for a three-dimensional line
        fig = plt.figure(2)
        fig.clf()
        ax = Axes3D(fig)
        x,y,z = zip(*coords)
        xgrid = np.linspace(np.min(x), np.max(x), num_episodes)
        ygrid = np.linspace(np.min(y), np.max(y), num_episodes)
        zgrid = np.linspace(np.min(z)-1, np.max(z)+1, num_episodes)
        #ax.plot3D(xgrid, ygrid, zgrid, 'gray')

        # Data for three-dimensional scattered points
        index_lastTen = max(0,len(x)-10)
        
        ax.scatter(np.array(x)[:index_lastTen], 
                   np.array(y)[:index_lastTen], 
                   np.array(z)[:index_lastTen], 
                   zdir=z, c='g')
        
        ax.scatter(np.array(x)[index_lastTen:], 
                   np.array(y)[index_lastTen:], 
                   np.array(z)[index_lastTen:], 
                   zdir=z, c='r')
        
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_zlabel('z position')
        
        fig = plt.figure(3)
        fig.clf()
        plt.plot(np.array(x), label='x', marker = '.')
        plt.plot(np.array(y), label='y', marker = '.')
        plt.plot(np.array(z), label='z', marker = '.')
        plt.legend()
        _ = plt.ylim()
        
        #print("z = ", np.array(z))
    print("Performance (average over last 10 rewards) = ", sum(rewards[index_lastTen:])/10)