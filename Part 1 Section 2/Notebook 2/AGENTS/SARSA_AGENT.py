# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


def get_next_state(s,action):

        #
        # Given a state and action, return the next state (location on the grid)
        #
        if s=='A' and action=='Right':
            action='B'
        elif s=='A' and action=='Down':
            action='E'
        elif s=='B' and action=='Right':
            action='C'
        elif s=='B' and action=='Left':
            action='A'
        elif s=='B' and action=='Down':
            action='F'
        elif s=='C' and action=='Right':
            action='D'
        elif s=='C' and action=='Left':
            action='B'
        elif s=='C' and action=='Down':
            action='G'
        elif s=='D' and action=='Left':
            action='C'
        elif s=='D' and action=='Down':
            action='H'
        elif s=='E' and action=='Up':
            action='A'
        elif s=='E' and action=='Right':
            action='F'
        elif s=='E' and action=='Down':
            action='I'
        elif s=='F' and action=='Up':
            action='B'
        elif s=='F' and action=='Down':
            action='J'
        elif s=='F' and action=='Right':
            action='G'
        elif s=='F' and action=='Left':
            action='E'        
        elif s=='G' and action=='Up':
            action='C'
        elif s=='G' and action=='Down':
            action='K'
        elif s=='G' and action=='Right':
            action='H'
        elif s=='G' and action=='Left':
            action='F'   
        elif s=='H' and action=='Up':
            action='D'
        elif s=='H' and action=='Down':
            action='L'
        elif s=='H' and action=='Left':
            action='G'
        elif s=='I' and action=='Up':
            action='E'  
        elif s=='I' and action=='Down':
            action='M'
        elif s=='I' and action=='Right':
            action='J'
        elif s=='J' and action=='Up':
            action='F'
        elif s=='J' and action=='Down':
            action='N'  
        elif s=='J' and action=='Right':
            action='K'
        elif s=='J' and action=='Left':
            action='I'
        elif s=='K' and action=='Up':
            action='G'
        elif s=='K' and action=='Down':
            action='O'  
        elif s=='K' and action=='Right':
            action='L'
        elif s=='K' and action=='Left':
            action='J'
        elif s=='L' and action=='Up':
            action='H'
        elif s=='L' and action=='Down':
            action='P'  
        elif s=='L' and action=='Left':
            action='K'        
        elif s=='M' and action=='Up':
            action='I'
        elif s=='M' and action=='Right':
            action='N'
        elif s=='N' and action=='Up':
            action='J'  
        elif s=='N' and action=='Right':
            action='O'
        elif s=='N' and action=='Left':
            action='M'
        elif s=='O' and action=='Up':
            action='K'
        elif s=='O' and action=='Right':
            action='P'  
        elif s=='O' and action=='Left':
            action='N'
        elif s=='P' and action=='Up':
            action='L'
        elif s=='P' and action=='Left':
            action='O'
        
        next_state=action    
        return next_state


class SARSA_AGENT:
    def __init__(self, r_matrix=None, q_matrix=None, initial_state='A', terminal_state='P', discount_factor=0.5, learning_rate=0.5):
        
        self.R = r_matrix
        self.Q = q_matrix

        self.s = initial_state
        
        self.a=None
        
        self.s_end = terminal_state
        
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
    def get_actions(self, s):
        # list of possible actions in state s
        A = self.R.loc[s,self.R.loc[s] >= -100].index
        return A.tolist()
        
    def update_q_take_step(self, action, epsilon):
          
        prev_q = self.Q.loc[self.s,action]
        reward = self.R.loc[self.s,action]
        
        s_next = get_next_state(self.s,action) #s_next, is the state you end up by taking action s_next == a
        Actions_next = self.get_actions(s_next) #possible actions to take from this next state
        
               
        # E-greedy policy, same as behaviour policy, for SARSA IMPLEMENTATION
        
        if np.random.rand() < epsilon: 
            
            #options=[self.Q.loc[s_next, a_next] for a_next in A_next] #what are the options from this new state?

            #grab the action here, which action was chosen in order to get the nxt_q?
            nxt_a =np.random.choice(Actions_next)
            #nxt_q  = np.random.choice([self.Q.loc[s_next, a_next] for a_next in A_next]) #update the previous   Q value using a random q.
            nxt_q=[self.Q.loc[s_next, nxt_a]][0]
            #print('Options',options)
            #print('Random Q chosen ', max_q)
        else:
            #print(A_next)
            #options=[self.Q.loc[s_next, a_next] for a_next in A_next] #what are the options from this new state?
            max_index  = np.argmax([self.Q.loc[s_next, a_next] for a_next in Actions_next])
            nxt_a  = Actions_next[max_index]
            nxt_q  = np.max([self.Q.loc[s_next, a_next] for a_next in Actions_next]) #update the previous   Q value using the max Q of this new state.
            #print('Options',options)
            #print('Max Q chosen  ', max_q)
        
        
        
        # Q update function
        new_q = prev_q + self.learning_rate * ((reward + (self.discount_factor * nxt_q)) - prev_q)
        
        self.Q.loc[self.s,action] = new_q
        
        #print('Q update function | Next state -->', s_next)
        
        return  s_next, reward, nxt_a
        
    def step(self, epsilon=1):
        #print('We are in state -->', self.s)
        # possible actions in state s
        Actions = self.get_actions(self.s)
        #print('Possible actions in state', self.s, '-->', A_t)
        # choose action        
        
        if self.a==None:
              if np.random.rand() < epsilon: 
                    action = np.random.choice(Actions)
              else:
                  action = self.Q.loc[self.s].idxmax()
        
        else:
              action=self.a
        # perform q-update
        #print('The action the agent chooses to take is -->', a)
        new_state, reward, new_state_action  = self.update_q_take_step(action,epsilon) 
        

        # get new state observation
        self.s = new_state
        #get the action to be performed on the new state 
        self.a= new_state_action
        #print('The new state is -->', self.s)
        return reward
        
        
    def run(self, max_steps, epsilon=1):  
        #print('E==', epsilon)
        steps = 0
        
        initial_state=self.s
        #self.s=random.choice(self.Q.index) #set random starting state for each episode, uncomment this to start an episode from a random state

        rewards=[] # all rewards for 1 episode
        
        while True:
            reward=self.step(epsilon=epsilon)
            rewards.append(reward)
            steps += 1
            if self.s == self.s_end or steps >= max_steps:
                self.s=initial_state 
                self.a=None
                break
                
        #print('steps',steps)     
        
        return steps, np.array(rewards).sum()  





    def test_run(self,test_start_location, test_max_steps, test_epsilon=0.0):  
        #print('E==', test_epsilon)
        #print('Max steps',test_max_steps)
        #print('Test location',test_location)
        
        test_steps = 0
        self.s=test_start_location  #Location that the Agent will start from
        
        test_rewards=[]
        while True:
            
            reward=self.step(epsilon=test_epsilon)
            test_rewards.append(reward)
            
            test_steps += 1
            #print('number of steps ',test_steps)
            #print('number of rewards',len(test_rewards))


            if self.s == self.s_end or test_steps >= test_max_steps: #stop if terminal state is reached, or if max steps are reached
                break
                
        #print(' final num of steps taken',test_steps)     
        
        return test_steps, test_rewards  
                
