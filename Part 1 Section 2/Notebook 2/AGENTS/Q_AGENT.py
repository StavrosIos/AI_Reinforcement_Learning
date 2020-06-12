# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:42:15 2020

@author: billy
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


class Q_Agent:
    
    def __init__(self, r_matrix=None, q_matrix=None, initial_state='A', terminal_state='P', discount_factor=0.5, learning_rate=0.5):
        
        # R matrix
        self.R = r_matrix
        
        # Q matrix
        self.Q = q_matrix
        
        # initial state
        self.s = initial_state
        
        # terminal state
        self.s_end = terminal_state
        
        #Γ factor
        self.discount_factor = discount_factor
        
        # α or η 
        self.learning_rate = learning_rate
        
    def get_actions(self, s):
        # list of possible actions in state s
        A = self.R.loc[s, self.R.loc[s] >= -100].index #all legal actions (exclusing NaNs), all actions as well as harmful ones 
        return A.tolist()
        
    def update_q_take_step(self, action):
        "Update Q value using the Bellman equation for Q-Learning"
        
        prev_q = self.Q.loc[self.s,action]
        reward = self.R.loc[self.s,action]
        
        s_next = action 
        state_next = get_next_state(self.s,s_next)
        Actions_next = self.get_actions(state_next) #get possible actions from the next state
        
        #options=[self.Q.loc[state_next, action_next] for action_next in Actions_next]
        #print(options)
        
        max_q  = np.max([self.Q.loc[state_next, action_next] for action_next in Actions_next])      
        #print('Chosen q', max_q )
        
        # Q update function
        new_q = prev_q + self.learning_rate * ((reward + (self.discount_factor * max_q)) - prev_q)
        
        self.Q.loc[self.s,action] = new_q
        #print('Q update function | Next state -->', s_next)
        
        return state_next, reward
        
    def step(self, epsilon=1):
        
        #print('We are in state -->', self.s)
        # possible actions in state s
        actions = self.get_actions(self.s)
        #print('Possible actions in state', self.s, '-->', A_t)

        if np.random.rand() <= epsilon: #Choose randomly depending on e
            a = np.random.choice(actions)
        else:
            a = self.Q.loc[self.s].idxmax() #else choose action with the MaxQ
        
        #perform q-update
        #print('The action the agent chooses to take is -->', a)
        new_state, step_reward = self.update_q_take_step(a)
        
        # get new state observation
        self.s = new_state
        #print('The new state is -->', self.s)
        
        return step_reward


    #Function to test the AGENT and the optimal path 
    def test_run(self,test_location, test_max_steps, test_epsilon=0.00):  
        #print('E==', test_epsilon)
        #print('Max steps',test_max_steps)
        #print('Test location',test_location)
        
        test_steps = 0
        self.s=test_location  #Location that the Agent will start from
        
        test_rewards=[]
        
        while True:
            
            reward=self.step(epsilon=test_epsilon) #take a step in the env
            test_rewards.append(reward)
            
            test_steps += 1
            #print('number of steps ',test_steps)
            #print('number of rewards',len(test_rewards))

            if self.s == self.s_end or test_steps >= test_max_steps: #stop if terminal state is reached, or if max steps are reached
                self.s=test_location
                break
                
        #print(' final num of steps taken',test_steps)     
        
        return test_steps, test_rewards  
        
    def run(self, max_steps, epsilon=1):  #1 run() == 1 episode  
        #print('New episode with e==', epsilon)
        
        steps = 0
        #self.s=random.choice(self.Q.index) #set random starting state for each episode, uncomment this to start an episode from a random state
        initial_state=self.s

        rewards=[]
        
        while True:
            reward=self.step(epsilon=epsilon) # take step in the environment
            rewards.append(reward)
            steps += 1
            if self.s == self.s_end or steps >= max_steps: #stop if terminal state is reached, or if max steps are reached
                self.s=initial_state 
                break  
        
        return steps, np.array(rewards).sum()  #return steps for 1 episode and total reward for 1 episode
    
          
