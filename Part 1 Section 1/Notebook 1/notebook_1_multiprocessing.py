# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('AGENTS')    
import multiprocessing
  

S = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'] 

A = ['Up', 'Down', 'Left', 'Right']
R = [[np.NaN, 0, np.NaN, 0],
   [np.NaN, -100, 0, 0],
   [np.NaN, 0, 0, 0],
   [np.NaN, -100, 0, np.NaN],
   [0, 0, np.NaN, -100],
   [0, 0, 0, 0],
   [0, 0, -100, -100],
   [0, -100, 0, np.NaN],
   [0, -100, np.NaN, 0],
   [-100, 0, 0, 0],
   [0, 0, 0, -100],
   [-100, 100, 0, np.NaN],
   [0, np.NaN, np.NaN, 0],
   [0, np.NaN, -100, 0],
   [0, np.NaN, 0, 100],
   [-100, np.NaN, 0, np.NaN]]
  
R = pd.DataFrame(columns=A,index=S,data=R)

Q = [[np.NaN, 0, np.NaN, 0],
[np.NaN, 0, 0, 0],
[np.NaN, 0, 0, 0],
[np.NaN, 0, 0, np.NaN],
[0, 0, np.NaN, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, np.NaN],
[0, 0, np.NaN, 0],
[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, np.NaN],
[0, np.NaN, np.NaN, 0],
[0, np.NaN, 0, 0],
[0, np.NaN, 0, 0],
[0, np.NaN, 0, np.NaN]]

Q = pd.DataFrame(columns=A,index=S,data=Q) 





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
  
      

def test_agent(R,learned_Q,start_location,max_steps):
  test_agent=Q_Agent(R,learned_Q,'A','P') #create a trained agent (Learned Q)

  stps, rewards = test_agent.test_run(start_location,max_steps,0)#execute an episode to find the terminal state P with 0 epsilon


  return stps, rewards





def metrics_steps_per_x_episodes(episodes_steps,max_steps,agent):
      
      title=None
      if agent=='Q':
          title="Sresults-Q_Agent.png"
      elif agent=='S':
           title="Sresults-Sarsa_Agent.png"
      elif agent=='R':
           title="Sresults-Random_Agent.png"

      
      steps_per_x_episodes=[]
      j=0
      for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]:
          steps_per_x_episodes.append(np.array(episodes_steps[j:i]).sum())
          j=i             
          
      plt.plot(steps_per_x_episodes[:])  
      plt.title('Steps per batch of episodes')
      plt.ylabel('Steps') 
      plt.xlabel('Episodes per batch')         
      plt.savefig('multiprocessing results/'+title)
      plt.show()



def metrics_reward_per_x_episodes(episodes_rewards,max_steps,agent):
      
      
      title=None
      if agent=='Q':
          title="Rresults-Q_Agent.png"
      elif agent=='S':
           title="Rresults-Sarsa_Agent.png"
      elif agent=='R':
           title="Rresults-Random_Agent.png"     

      rewards_per_x_episodes=[]
      j=0
      for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]:
          rewards_per_x_episodes.append(np.array(episodes_rewards[j:i]).sum())
          j=i       

      plt.plot(rewards_per_x_episodes[:])  
      plt.title('Reward per batch of episodes')
      plt.ylabel('Reward') 
      plt.xlabel('Episodes per batch')       
      plt.savefig('multiprocessing results/'+ title)
      plt.show()





def metrics_percentages_per_batch(episodes_rewards,max_steps,agent):
      
      
      title=None
      if agent=='Q':
          title="Presults-Q_Agent.png"
      elif agent=='S':
           title="Presults-Sarsa_Agent.png"
      elif agent=='R':
           title="Presults-Random_Agent.png"     

      sucessfull_percentages_per_batch=[]
      failed_percentages_per_batch=[]
      
      j=0
      for i in range (len(episodes_rewards)):
      
        sucess_count=0
        fail_count=0
      
        if i%1000==0:
          s=np.array(episodes_rewards[j:i])
          for score in s:
            if score==100:
              sucess_count+=1
            else:
              fail_count+=1
      
          success_score=sucess_count/1000 #possible variable spoil here
          fail_score= fail_count/1000
      
          sucessfull_percentages_per_batch.append(success_score)
          failed_percentages_per_batch.append(fail_score)
      
          j=i


      plt.plot(sucessfull_percentages_per_batch[:])  
      plt.title(' Percentage of sucessful episodes per batch ')
      plt.ylabel('Sucess %')
      plt.xlabel('Episodes per batch')         
      plt.savefig('multiprocessing results/'+title)
      plt.show()
      



def metrcis_cummulative_rewards(episodes_rewards,agent):

      
      title=None
      if agent=='Q':
          title="Cresults-Q_Agent.png"
      elif agent=='S':
           title="Cresults-Sarsa_Agent.png"
      elif agent=='R':
           title="Cresults-Random_Agent.png"     

      cummulative_rewards=[]
      suma=0
      for reward in episodes_rewards:
          suma=suma+reward
          cummulative_rewards.append(suma)
      plt.plot(cummulative_rewards[1:])
      plt.title(' Cummulative reward')
      plt.savefig('multiprocessing results/'+title)
      plt.show()





def train_one():
      
      ini_state='A'
      term_state='P'
      
      q_agent = Q_Agent(R.copy(),Q.copy(),ini_state,term_state,discount_factor=0.5,learning_rate=0.5)
      
      #lets pass in the R and Q matrix as arguments, initial state,terminal state, and the 2 learning parameters
      
      decay = 0.9997
      
      min_epsilon = 0.05 # this is the minimum allowable epsilon, so the resulting policy is not 100% deterministic. 5% of actions will be random
      
      epsilon=1.0
      
      episodes_steps=[]
      
      episodes, max_steps = 20_000, 20
      
      episodes_rewards=[]
      
      
      for i in range(episodes):
          
          epsilon = max(min_epsilon, epsilon*decay) #pass a new epsilon that is more greedy as the number of episodes increases
          
          ep_stps, ep_reward = q_agent.run(max_steps,epsilon) # excute an episode, with decay
      
          episodes_rewards.append(ep_reward)    
          
          episodes_steps.append(ep_stps)
          
          if (i%1000==0):
              print('Batch completed')
              
      
      metrics_steps_per_x_episodes(episodes_steps,max_steps,'Q')
      
      
      metrics_reward_per_x_episodes(episodes_rewards,max_steps,'Q')
      
      
      
      metrics_percentages_per_batch(episodes_rewards,max_steps,'Q')
      
      
      all_locations=Q.index
      final=dict.fromkeys(all_locations,0)
      
      for location in all_locations:
        if location not in ['F','H','L','M','P']:
          results=test_agent(q_agent.R,q_agent.Q,location,20)
          final[location]=results
      
      print(final)
      
      
      results=test_agent(q_agent.R,q_agent.Q,'E',20)
      print("Path from "+"E",results)
      
      
      metrcis_cummulative_rewards(episodes_rewards,'Q')



def train_two():
      
      ini_state='A'
      term_state='P'
      
      random_agent=Q_Agent(R.copy(),Q.copy(),ini_state,term_state,discount_factor=0.5,learning_rate=0.5)
      
      #lets pass in the R and Q matrix as arguments, initial state,terminal state, and the 2 learning parameters
      
      epsilon=1.0
      
      episodes, max_steps = 20_000, 20 
      
      episodes_steps=[]
      
      episodes_rewards=[]
      
      
      for i in range(episodes):
              
          ep_stps, ep_reward = random_agent.run(max_steps,epsilon) # excute an episode, with decay
      
          episodes_rewards.append(ep_reward)    
          
          episodes_steps.append(ep_stps)
            
          if (i%1000==0):
              print('Batch completed')
              

      metrics_steps_per_x_episodes(episodes_steps,max_steps,'R')


      metrics_reward_per_x_episodes(episodes_rewards,max_steps,'R')
      
      
      metrics_percentages_per_batch(episodes_rewards,max_steps,'R')
      
            
      metrcis_cummulative_rewards(episodes_rewards,'R')

      



def train_three():
      
      ini_state='A'
      term_state='P'
      
      sarsa_agent = SARSA_AGENT(R.copy(),Q.copy(),ini_state,term_state,discount_factor=0.5,learning_rate=0.5)
      
      decay = 0.9997
      
      min_epsilon = 0.05 # this is the minimum allowable epsilon, so the resulting policy is not 100% deterministic. 5% of actions will be random
      
      epsilon=1.0
      
      episodes_steps=[]
      
      episodes, max_steps = 20_000, 20
      
      episodes_rewards=[]
      
      for i in range(episodes):
          
          epsilon = max(min_epsilon, epsilon*decay) #pass a new epsilon that is more greedy as the number of episodes increases
          
          ep_stps, ep_reward = sarsa_agent.run(max_steps,epsilon) 
      
          episodes_rewards.append(ep_reward)    
          
          episodes_steps.append(ep_stps)
      
      
          if (i%1000==0):
              print('Batch completed')
      
      
      
      metrics_steps_per_x_episodes(episodes_steps,max_steps,'S')

      metrics_reward_per_x_episodes(episodes_rewards,max_steps,'S')
      
      metrics_percentages_per_batch(episodes_rewards,max_steps,'S')
                  
      metrcis_cummulative_rewards(episodes_rewards,'S')




if __name__ == '__main__':
    
      p1=multiprocessing.Process(target=train_one)
      p2=multiprocessing.Process(target=train_two)
      p3=multiprocessing.Process(target=train_three)


      p1.start()
      p2.start()
      p3.start()

      p1.join()
      p2.join()
      p3.join()

      print('Finished')



