# AI_Reinforcement_Learning

Q-Learning, Sarsa, Deep Q-Learning, Markov Decision Problem, State-transition function, Reward function, ε-greedy policy

The Robot Waiter:
The domain chosen for this study was devised and developed by the authors of this report, who got inspired by the OpenAI Frozen Lake example. The owner of a restaurant has employed the use of a robot to deliver food to customers. The robot will have to learn by itself how to move through the environment by interacting with it. The best approach to that type of learning is to use Q-learning as the learning mechanism. To accomplish the task, the agent will have to find a policy. The icy restaurant that the greatest reward is gained in the minimum number of steps. The environment of the ice restaurant contains a total of 16 locations (4 by 4 grid). Some of the tiles are occupied by customers that do not want to be assaulted by the agent.

The Stock Market Trader:
To tackle stock trading using RL methods, the problem needs to be formalized as a Markov Decision Process. For example, if we have some observation about the price of a specific company stock, we want to know whether we should buy, sell, or do nothing. If the stock is bought before the price goes up, then we make profit, however, if we buy and the price goes down, we lose money. This simple framing of the problem can be tackled with RL techniques. Next we formalize the problem, so that an agent can be built to maximize profits. Three things are needed for developing an RL approach.
1. A state that is made up of observations from the environment. Each state should be distinguishable from all other states.
2. A set of possible actions for each state. In this setting all states have the same possible actions.
3. Lastly a suitable reward function. The reward function will guide the learning process for the agent.
