import trading_environment  #Trading enviornment class
import torch 
import torch.optim as optim 
import numpy as np 
from datetime import datetime #time the execution of the script
import matplotlib.pyplot as plt 
import my_nn_architectures #Deep Q netowrk architecture to use
import data_utility #load data
import helper_functions as helper


stock_data, series=data_utility.load_stock('Data/IBM_train_set.txt')  #lOAD THE DATA

# Environment Features 
# Reward is given by the environment when the Agent closes their position
# The state is characterized by the last 10 Bars in the time series
env = trading_environment.StocksEnv(stock_data, bars_count=10 , random_offset=True) 
##Deep Q network
device = torch.device("cuda") # "cpu" or "cuda" is used making sure CUDA is used to speed-up training     
#main net 
main_net = my_nn_architectures.DuelingDQN(env.observation_space.shape[0],env.action_space.n).to(device)
#target net
target_net = my_nn_architectures.DuelingDQN(env.observation_space.shape[0],env.action_space.n).to(device)
target_net.eval()


print(main_net)
print(target_net)
    
### Training loop ###
SAVES_PATH="saved agents"
REPLAY_SIZE=30_000
EPSILON_START=1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY = 50_000 
GAMMA=0.99
REPLAY_START_SIZE=2_000
SYNC_TARGET=4_000
BATCH_SIZE=64  #experiences to train the networks on!!!, alternate between 10 and 5 k and time it


## optimizer for the objective function
LEARNING_RATE = 1e-4
optimizer = optim.Adam(main_net.parameters(), lr=LEARNING_RATE)
#create the Agent, using the environment and buffer
buffer = helper.ReplayMemory(REPLAY_SIZE)


# Maximum episodes can be set here as a stoping criterion
epsilon = EPSILON_START
total_rewards=[] #holding rewards for each step in the environment
MAX_STEPS=60_000 
times_synched=0
steps=0

startTime = datetime.now() # time the training loop 
state=env.reset()
mean_rewards=[]
while(True):      
      
      steps+=1     
      epsilon = max(EPSILON_FINAL, EPSILON_START - steps / EPSILON_DECAY)      
      reward, state, done =helper.take_step(epsilon,state,main_net,env,buffer,'cuda')
      
      if len(buffer) < REPLAY_START_SIZE: #play with the environment until buffer is populated
            print("Buffer size ",len(buffer))
            print("Populating buffer")
            print(env.offset)
            continue      # go to the start of the loop

      if (done):  # continue here once the buffer is full
            total_rewards.append(reward) #for each episode
            m_reward = np.mean(total_rewards[-100:])
            mean_rewards.append(m_reward)
            print("steps %d: played %d games, mean_last_100_reward %.3f, "
            "eps %.2f" % (steps, len(total_rewards), m_reward, epsilon))   
           
      if steps % SYNC_TARGET == 0: #every x steps nets are sycnced
            print("Nets Synched")
            times_synched+=1
            target_net.load_state_dict(main_net.state_dict())   
      
            
      #else, train network, if buffer is big enough
      sample_batch = buffer.sample(BATCH_SIZE) # draw a sample of experiences      
      loss=my_nn_architectures.double_dqn_loss(sample_batch, main_net, target_net, gamma=GAMMA ,device=device)
      optimizer.zero_grad()# reduce all gradients to zero            
      loss.backward() #erorr back-prop      
      optimizer.step() # optimization step
      

      if steps == MAX_STEPS: #stop criterion for now...
            print("Training completed")
            name=("10_bars_DuelingDoubleDQN_OFS_T_mean_value-%.3f.data" % m_reward)
            path = SAVES_PATH +"/"+ name
            #torch.save(main_net.state_dict(), path)
            break
      
      
cummulative_rewards=[]
suma=0
for reward in total_rewards:
    suma=suma+reward
    cummulative_rewards.append(suma)
plt.plot(cummulative_rewards)
plt.title("Agent: Dueling Double DQN - Cummulative Reward")
plt.xlabel('Episode count')
#plt.savefig('Results/Training results/Dueling Double DQN-Training results.png')
plt.show()


print("Time taken ",datetime.now() - startTime) # save in notepad file
print("Times synched ",times_synched)  #wriong I think 



plt.plot(mean_rewards)
plt.title("Agent: Dueling DQN - Mean Reward")
#plt.savefig('Results/Training results/DuelingDQN-Training mean results.png')
plt.show()








