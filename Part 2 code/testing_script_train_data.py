"""
This script tests the trained agents on the test data.

"""

import trading_environment  #Trading enviornment class
import torch # Neural Network library
import numpy as np # as always
import matplotlib.pyplot as plt 
import my_nn_architectures
import data_utility


volumes = input("Which volume feature to use? Input 'absolute' or 'std' :")
print("Selected: " + volumes)

if volumes == 'absolute': 
      stock_data , series = data_utility.load_stock_absolute_volumes('Data/IBM_train_set.txt')  
      
elif volumes == 'std':
      stock_data , series = data_utility.load_stock('Data/IBM_train_set.txt')  

else: 
      raise Exception('input not of the correct format')
      

#Testing environment
testing_environment = trading_environment.StocksEnv(stock_data, series=series ,bars_count=10, random_offset=True)

#Load trained model from the directory
name=("10_bars_DQN_OFS_T_mean_value-%.3f.data" % 4.745) 
SAVES_PATH="saved agents"
trained_net_DQN = my_nn_architectures.DQN(testing_environment.observation_space.shape[0], testing_environment.action_space.n)
trained_net_DQN.load_state_dict(torch.load(SAVES_PATH+"/"+name)) 
trained_net_DQN.cuda()
trained_net_DQN.eval()

DoubleDQNname=("10_bars_DoubleDQN_OFS_T_mean_value-%.3f.data" % 4.657)
trained_net_DoubleDQN = my_nn_architectures.DQN(testing_environment.observation_space.shape[0], testing_environment.action_space.n)
trained_net_DoubleDQN.load_state_dict(torch.load(SAVES_PATH+"/"+DoubleDQNname)) 
trained_net_DoubleDQN.cuda()
trained_net_DoubleDQN.eval()

DuelingDoubleDQNname=("10_bars_DuelingDoubleDQN_OFS_T_mean_value-%.3f.data" % 4.588) 
trained_net_DuelingDQN = my_nn_architectures.DuelingDQN(testing_environment.observation_space.shape[0], testing_environment.action_space.n)
trained_net_DuelingDQN.load_state_dict(torch.load(SAVES_PATH+"/"+DuelingDoubleDQNname)) 
trained_net_DuelingDQN.cuda()
trained_net_DuelingDQN.eval()


device = torch.device("cuda")

print(trained_net_DQN)
print(trained_net_DoubleDQN)
print(trained_net_DuelingDQN)

f_state = testing_environment.reset()


rewards=[] #for each episode
total_reward=0
index=0

print("Start DQN testing")

test_episodes=2_000
test_index=0
total_reward=0

state=f_state
while (test_index < test_episodes):
      
 
      state_tensor = torch.tensor([state]).to(device)
      out_v = trained_net_DQN(state_tensor)
      action_idx = out_v.max(dim=1)[1].item() #action with highest Q value.
      action = trading_environment.Actions(action_idx)
      
      new_state, reward, done, info = testing_environment.step(action)
      
      state=new_state
      
      if done: #if episode is completed, then save the reward
            test_index+=1
            rewards.append(reward)
            total_reward += reward
            testing_environment.reset() #reset for the next episode
            
            if test_index % 1_000 == 0: #print total reward every x episodes
                  print('DQN testing')
                  print("%d: reward=%.3f" % (test_index, total_reward))
      

cummulative_rewards=[]
suma=0
for reward in rewards:
    suma=suma+reward
    cummulative_rewards.append(suma)
plt.plot(cummulative_rewards)
plt.title("Cummulative Reward")
plt.xlabel('Episode count')
plt.ylabel(" Reward")
plt.show()


      
dqn_c_rewards=cummulative_rewards.copy()





###Random agent testing
random_rewards=[]
test_index=0
total_reward=0
print("Start Random Agent testing")


while (test_index < test_episodes):      
      act = trading_environment.Actions(np.random.randint(0,3))       
      action = trading_environment.Actions(act)
      
      new_state, reward, done, info = testing_environment.step(action) #steps do not have rewards      
      state=new_state      
      
      if done: 
            test_index+=1
            #print("Episode completed", test_index)  
            random_rewards.append(reward)
            total_reward += reward
            testing_environment.reset() 
            
            if test_index % 1_000 == 0: 
                  print('Ranndom testing')
                  print("%d: rewar6d=%.3f" % (test_index, total_reward))
      

cummulative_rewards=[]
suma=0
for reward in random_rewards:
    suma=suma+reward
    cummulative_rewards.append(suma)
plt.plot(cummulative_rewards)
plt.title("Cummulative Reward")
plt.xlabel('Episode count')
plt.ylabel(" Reward")
plt.show()

random_c_rewards=cummulative_rewards.copy()



#==========================
#=========================
#===========================





###Double DQN 
f_state = testing_environment.reset()


print("DOUBLE DQN testing")
DoubleDQN_rewards=[]
test_index=0
total_reward=0
state=f_state

while (test_index < test_episodes):
      
      
      state_tensor = torch.tensor([state]).to(device) 
      out_v = trained_net_DoubleDQN(state_tensor)
      action_idx = out_v.max(dim=1)[1].item() #action with highest Q value.
      action = trading_environment.Actions(action_idx)      
       
      new_state, reward, done, info = testing_environment.step(action) #steps do not have rewards
      
      state=new_state
      
      if done: #if episode is completed, then save the reward
            test_index+=1

            DoubleDQN_rewards.append(reward)
            
            total_reward += reward
            testing_environment.reset() #reset for the next episode
            
            if test_index % 1_000 == 0: 
                  print('Double DQN testing')
                  print("%d: reward=%.3f" % (test_index, total_reward))



cummulative_rewards=[]
suma=0
for reward in DoubleDQN_rewards:
    suma=suma+reward
    cummulative_rewards.append(suma)
plt.plot(cummulative_rewards)
plt.title("Cummulative Reward")
plt.xlabel('Episode count')
plt.ylabel(" Reward")
plt.show()

DoubleDQN_c_rewards=cummulative_rewards.copy()



plt.plot(dqn_c_rewards,label='DQN Agent')
plt.plot(random_c_rewards,label='Random Agent')
plt.plot(DoubleDQN_c_rewards,label='Double DQN Agent')
plt.legend()
plt.show()



#==========================
#=========================
#===========================



### DUELING DQN
print(" Dueling DOUBLE DQN testing")
DuelingDQN_rewards=[]
test_index=0
total_reward=0
state=f_state

while (test_index < test_episodes):
      
      state_tensor = torch.tensor([state]).to(device)
      out_v = trained_net_DuelingDQN(state_tensor)
      action_idx = out_v.max(dim=1)[1].item() #action with highest Q value.
      action = trading_environment.Actions(action_idx)
      new_state, reward, done, info = testing_environment.step(action) #steps do not have rewards
      
      state=new_state
      
      if done: #if episode is completed, then save the reward
            test_index+=1

            #print("Episode completed", test_index)  
            DuelingDQN_rewards.append(reward)
            
            total_reward += reward
            testing_environment.reset() #reset for the next episode
            
            if test_index % 1_000 == 0:
                  print('Dueling Double DQN testing')
                  print("%d: reward=%.3f" % (test_index, total_reward))
      

cummulative_rewards=[]
suma=0
for reward in DuelingDQN_rewards:
    suma=suma+reward
    cummulative_rewards.append(suma)
plt.plot(cummulative_rewards)
plt.title("Cummulative Reward")
plt.xlabel('Episode count')
plt.ylabel(" Reward")
plt.show()


DuelingDoubleDQN_c_rewards=cummulative_rewards.copy()


###
###
#PLOT ALL REWARDS
### 

#Plotting Total Reward 
plt.plot(dqn_c_rewards,label='DQN Agent')
plt.plot(random_c_rewards,label='Random Agent')
plt.plot(DoubleDQN_c_rewards,label='Double DQN Agent')
plt.plot(DuelingDoubleDQN_c_rewards,label='Dueling Doule DQN')
plt.title("Cummulative Reward (Training Data)")
plt.xlabel('Episode count')
plt.ylabel(" Reward")
plt.legend()
plt.savefig('Results/Training results/Profit on training data.png')
plt.show()

















