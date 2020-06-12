"""
This script contains code that is needed for training the agents.
In order to to de-clutter the main training scripts, this script is used.

"""

import torch
import torch.nn as nn
import numpy as np
import collections


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])

#Step function 
#Saves transition in the memory replay buffer
#Takes  a step in the environment, returns the reward,new_state and Done flag
@torch.no_grad()
def take_step(e,state,main_net,env,replay_buffer,device):
      
      step_reward=0.0
      action=select_action(e,state,env,device,main_net)
      
      # take step in the environment
      new_state, stp_r, done, _ = env.step(action)
      step_reward = stp_r
      
      #create Experience and store in buffer, state made up of 10 past bars
      exp = Experience(state.reshape(42), action, step_reward, done, new_state.reshape(42))
      replay_buffer.push(exp)
      
      if done: #if done=True then agent has completed a game
            new_state=env.reset()          
  
      return step_reward,new_state,done



      


def select_action(e,state,env,device,main_net):
      action=None
      with torch.no_grad():
            
            if np.random.random() >= e:
                  
                  state = np.array([state], copy=False)
                  state_tensor = torch.tensor(state).to(device) #create pytorch tensor for state
                  action_v=main_net(state_tensor).max(1)[1].view(1, 1)
                  action=int(action_v.item())
                  
            else:
                 
                  action = env.action_space.sample() #random action to take 
      

      return action             



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states, actions, rewards, dones, next_states = zip(*[self.memory[index] for index in indices])

        return np.array(states,dtype=np.float32), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


    def __len__(self):
        return len(self.memory)

















