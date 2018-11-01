# Project: Navigation

The aim of the project is to train an agent to navigate and collect yellow bananas while avoiding blue bananas in a large square world. Project uses unity-ml agents(a library which provides different environments for training and testing various reinforcement learning algorithms) and torch(deep learning library).

## Environment 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. A trained agent will collect as many yellow bananas as possible while avoiding blue bananas.There are four actions agent can take,

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The problem is episodic and agent must get an average score of +13 over 100 consecutive episodes to solve the environment 

## Learning Algorithm

The environment is solved using deep Q-network, a reinforcement learning algorithm(DQN).The input to the deep neural network is the state vector and the output will be an action. DQN uses Experience Replay and Target network to get stable and improved performace.

### Deep learning model architecture:-<br>


    Input(state vector[37])
  
    Fully connected(64)
  
          Relu
  
    Fully connected(64)
  
          Relu
  
    output(action[4])

### Hyperparameters:-
 --> mini batch_size = 64<br>
 --> discount factor(gamma) = 0.99<br>
 --> tau = 0.001<br>
 --> update_steps = 4<br>
 --> optimizer = Adam<br>
 --> learning_rate = 5e-4<br>
 
## Plot of score vs episodes(training)
![alt text](https://github.com/Nishanth009/Udacity-Nano-Degree-RL-Project-Navigation/blob/master/images/plot_rewards.png)

Weights file **checkpoint.pth**

## Ideas for Future Work

1) Upgarde DQN with prioritized experience replay, Dueling DQN, Noisy DQN
2) Learn from pixels -- images/frames as input and action as output
3) Try various hyperparameters and observe how it effects the agents performance


