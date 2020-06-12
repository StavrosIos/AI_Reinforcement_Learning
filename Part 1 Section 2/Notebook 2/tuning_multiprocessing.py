# -*- coding: utf-8 -*-


import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt


import pandas as pd
import sys

sys.path.append('AGENTS')    
from Q_AGENT import Q_Agent
from SARSA_AGENT import SARSA_AGENT
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
  

def train_1():
      
    print('ALPHA TUNING-SARSA')  

   
    alphas = [ 0.1, 0.5, 0.8] # Train a classifer, each one with different Alpha learning rates. (Train them sequentially)
    all_metrics_per_alpha=[]
    
    for alpha in alphas:

          min_epsilon = 0.05
          epsilon_new=1
          decay=0.9997

          print('Starting epsilon ',epsilon_new) 
          #the starting epsilon should go back to 1 for every classifier training loop


          sarsa_agent = SARSA_AGENT(R.copy(),Q.copy(),'A','P',discount_factor=0.5, learning_rate=alpha)  #To pass in the learning rate, we need to pass it to the class definition 

          episodes, max_steps = 10_000, 20 
          all_rewards=[]

          for i in range(episodes):

              if (i==episodes-1):
                  print("\n LAST EPISODE \n")

              epsilon_new = max(min_epsilon, epsilon_new*decay) #pass a new epsilon that is more greedy as the number of episodes increases

              stps, rewards = sarsa_agent.run(max_steps,epsilon_new) # excute an episode, with decay

              all_rewards.append(np.array(rewards).sum())
            
              if (i%1000==0):
                    print('Batch ended')

          rewards_per_x_episodes=[]
          j=0
          for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
            rewards_per_x_episodes.append(np.array(all_rewards[j:i]).sum())
            j=i

          all_metrics_per_alpha.append(rewards_per_x_episodes)
          plt.plot(rewards_per_x_episodes[:],label="alpha= "+str(alpha)+"/ gamma= "+str(0.5)+"/ decay= "+str(decay))
          plt.legend()  
          plt.title('Alpha Tuning SARSA ')
          plt.ylabel('Reward') 
          plt.xlabel('Episodes per batch of X')      
          plt.savefig('multiprocessing results/SARSA-ALPHA TUNING.png',bbox_inches = "tight")
          print(rewards_per_x_episodes)

    plt.show()
    


def train_2():
      
    print('GAMMA TUNING-SARSA')  
    
    
    gammas = [ 0.1, 0.5, 0.8] 
    all_metrics_per_gamma=[]
    for gamma in gammas:


          min_epsilon = 0.05
          epsilon_new=1
          decay=0.9997 

          print('Starting epsilon ',epsilon_new) 

          sarsa_agent = SARSA_AGENT(R.copy(),Q.copy(),'A','P',discount_factor=gamma, learning_rate=0.5)

          all_rewards=[]
          episodes, max_steps = 10_000, 20 

          for i in range(episodes):

              if (i==episodes-1):
                  print("\n LAST EPISODE \n")

              epsilon_new = max(min_epsilon, epsilon_new*decay) #pass a new epsilon that is more greedy as the number of episodes increases

              stps, rewards = sarsa_agent.run(max_steps,epsilon_new) # excute an episode, with decay

              all_rewards.append(np.array(rewards).sum())
                
              if (i%1000==0):
                    print('Batch ended') 



          rewards_per_x_episodes=[]
          j=0
          for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
            rewards_per_x_episodes.append(np.array(all_rewards[j:i]).sum())
            j=i           


          all_metrics_per_gamma.append(rewards_per_x_episodes)
          plt.plot(rewards_per_x_episodes[:],label="gamma= "+str(gamma)+"/alpha ="+str(0.5)+"/decay= "+str(decay))
          plt.legend()    
          plt.title('Gamma Tuning SARSA')
          plt.ylabel('Reward') 
          plt.xlabel('Episodes per batch')      
          plt.savefig('multiprocessing results/SARSA- GAMMA TUNING.png',bbox_inches = "tight")

    plt.show()



def train_3():
          
      
      print('ALPHA TUNING-Q-AGENT')  
      
      
      alphas = [0.1, 0.5, 0.8] 
      all_metrics_per_alpha=[]
      
      for alpha in alphas:
      
            min_epsilon = 0.05
            epsilon_new=1
            decay=0.9997
          
            print('Starting epsilon ',epsilon_new) 
      
            q_agent = Q_Agent(R.copy(),Q.copy(),'A','P',discount_factor=0.5, learning_rate=alpha)  
      
            episodes, max_steps = 10_000, 20 
      
            all_rewards=[]
            
            for i in range(episodes):
                
                if (i==episodes-1):
                    print("\n LAST EPISODE \n")
                    
                epsilon_new = max(min_epsilon, epsilon_new*decay) #pass a new epsilon that is more greedy as the number of episodes increases
                
                stps, rewards = q_agent.run(max_steps,epsilon_new) # excute an episode, with decay
      
                all_rewards.append(np.array(rewards).sum())
                
                if (i%1000==0):
                    print('Batch ended')  
      
            

            rewards_per_x_episodes=[]
            j=0
            for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
              rewards_per_x_episodes.append(np.array(all_rewards[j:i]).sum())
              j=i
      
            all_metrics_per_alpha.append(rewards_per_x_episodes)
            plt.plot(rewards_per_x_episodes[:],label="alpha= "+str(alpha)+" /gamma= "+str(0.5)+" /decay= "+str(decay))
            plt.legend() 
            plt.title('Alpha Tuning Q-Agent')
            plt.ylabel('Reward') 
            plt.xlabel('Episodes per batch of X')
            plt.savefig('multiprocessing results/ Q-AGENT- ALPHA TUNING.png',bbox_inches = "tight")
              
      plt.show()




def train_4():
      
      print('GAMMA TUNING-Q-AGENT')  
      
      gammas = [ 0.1, 0.5, 0.8]
      all_metrics_per_gamma=[]
      
      
      for gamma in gammas:
      
            min_epsilon = 0.05
            epsilon_new=1
            decay=0.9997   
            
            print('Starting epsilon ',epsilon_new) 
      
            q_agent = Q_Agent(R.copy(),Q.copy(),'A','P',discount_factor=gamma, learning_rate=0.5)  #learning rate passed here would be the best performing one
      
            all_rewards=[]
      
            episodes, max_steps = 10_000, 20 
      
            for i in range(episodes):
                
                if (i==episodes-1):
                    print("\n LAST EPISODE \n")
                    
                epsilon_new = max(min_epsilon, epsilon_new*decay) #pass a new epsilon that is more greedy as the number of episodes increases
                
                stps, rewards = q_agent.run(max_steps,epsilon_new) # excute an episode, with decay
      
                all_rewards.append(np.array(rewards).sum())
      
                if (i%1000==0):
                    print('Batch ended')  
      


      
            rewards_per_x_episodes=[]
            j=0
            for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
              rewards_per_x_episodes.append(np.array(all_rewards[j:i]).sum())
              j=i  
      
            all_metrics_per_gamma.append(rewards_per_x_episodes)
            plt.plot(rewards_per_x_episodes[:],label="gamma= "+str(gamma)+" /alpha= "+str(0.5)+" /decay= "+str(decay))
            plt.legend()  
            plt.title('GAMMA TUNING-Q-AGENT ')
            plt.ylabel('Reward') 
            plt.xlabel('Episodes per batch of X')
            plt.savefig('multiprocessing results/ Q-AGENT- GAMMA TUNING.png',bbox_inches = "tight")
              
      plt.show()









def train_5():
      
      print(' Q-AGENT FINAL TUNING')
      
      alphas=[0.5,0.8]
      gammas=[0.5,0.8]
      decays=[ 0.9997, 0.9998, 0.9995]

      all_metrics_per_configuration=[] # the len of this list should be 8
      
      for alpha in alphas:
        for gamma in gammas:
          for decay in decays:
      
            min_epsilon = 0.05
            epsilon_new=1
            
            print('Starting epsilon ',epsilon_new) 
      
            q_agent = Q_Agent(R.copy(),Q.copy(),'A','P',discount_factor=gamma, learning_rate=alpha)  #learning rate passed here would be the best performing one
      
            all_rewards=[]
      
            episodes, max_steps = 10_000, 20 
      
      
            for i in range(episodes):
                
                if (i==episodes-1):
                    print("\n LAST EPISODE \n")
                    
                epsilon_new = max(min_epsilon, epsilon_new*decay) #pass a new epsilon that is more greedy as the number of episodes increases
                
                stps, rewards = q_agent.run(max_steps,epsilon_new) # excute an episode, with decay
      
                #print(epsilon_new)
      
                all_rewards.append(np.array(rewards).sum())
      

            rewards_per_x_episodes=[]
            j=0
            for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
              rewards_per_x_episodes.append(np.array(all_rewards[j:i]).sum())
              j=i   
      
            all_metrics_per_configuration.append(rewards_per_x_episodes)
            plt.plot(rewards_per_x_episodes[:],label="γ= "+str(gamma)+'/α= '+str(alpha)+" /d= "+str(decay))
            plt.legend()  
            plt.title(' Q-AGENT PARAMETER TUNING')
            plt.ylabel('Reward') 
            plt.xlabel('Episodes per batch')
            plt.savefig('multiprocessing results/ Q-AGENT- FINAL TUNING.png',bbox_inches = "tight")
              
      plt.show()
      




def train_6():
      
      print(' SARSA FINAL TUNING')
      
      alphas=[0.5,0.8]
      gammas=[0.5,0.8]
      decays=[ 0.9997, 0.9998, 0.9995] 
      
      all_metrics_per_configuration=[] # the len of this list should be 8
      
      for alpha in alphas:
        for gamma in gammas:
          for decay in decays:
      
            min_epsilon = 0.05
            epsilon_new=1
            
            print('Starting epsilon ',epsilon_new) 
      
            sarsa_agent = SARSA_AGENT(R.copy(),Q.copy(),'A','P',discount_factor=gamma, learning_rate=alpha)  #learning rate passed here would be the best performing one
      
            all_rewards=[]
      
            episodes, max_steps = 10_000, 20 
      
      
            for i in range(episodes):
                
                if (i==episodes-1):
                    print("\n LAST EPISODE \n")
                    
                epsilon_new = max(min_epsilon, epsilon_new*decay) #pass a new epsilon that is more greedy as the number of episodes increases
                
                stps, rewards = sarsa_agent.run(max_steps,epsilon_new) # excute an episode, with decay
            
                all_rewards.append(np.array(rewards).sum())

            rewards_per_x_episodes=[]
            j=0
            for i in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
              rewards_per_x_episodes.append(np.array(all_rewards[j:i]).sum())
              j=i   
      
            all_metrics_per_configuration.append(rewards_per_x_episodes)
            plt.plot(rewards_per_x_episodes[:],label="γ= "+str(gamma)+"/α= "+str(alpha)+"/d="+str(decay))
            plt.legend()  
            plt.title('SARSA PARAMETER TUNING')
            plt.ylabel('Reward') 
            plt.xlabel('Episodes per batch')
            plt.savefig('multiprocessing results/ SARSA- FINAL TUNING.png',bbox_inches = "tight")
              
      plt.show()









if __name__ == '__main__':
    
      p1=multiprocessing.Process(target=train_1)
      p2=multiprocessing.Process(target=train_2)
      p3=multiprocessing.Process(target=train_3)
      p4=multiprocessing.Process(target=train_4)
      p5=multiprocessing.Process(target=train_5)
      p6=multiprocessing.Process(target=train_6)

      p1.start()
      p2.start()
      p3.start()
      p4.start()
      p5.start()
      p6.start()
      
      p1.join()
      p2.join()
      p3.join()
      p4.join()
      p5.join()
      p6.join()
      
      print('Finished')


