# Project report


## Goal


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas. In order to solve the environment, our agent must achieve an average score of +13 over 100 consecutive episodes.

## Learning Algorithm 

To solve this project, I implemented Q-learning algorithm by using Deep Q network (DQN), with a target network and an experience replay buffer. I used a target network variant called [Double DQN learning](https://arxiv.org/pdf/1509.06461.pdf). Details of each of these methods, along with hyperparameters used, are given below.

### Deep Q Network (DQN)
Although Q-learning is a very powerful algorithm, its main weakness is lack of generality. If you view Q-learning as updating numbers in a two-dimensional array (Action Space * State Space), it, in fact, resembles dynamic programming. This indicates that for states that the Q-learning agent has not seen before, it has no clue which action to take. In other words, Q-learning agent does not have the ability to estimate value for unseen states. To deal with this problem, DQN get rid of the two-dimensional array by introducing Neural Network.
DQN leverages a Neural Network to estimate the Q-value function. The input for the network is the current, while the output is the corresponding Q-value for each of the action.
In 2013, DeepMind applied DQN to Atari game, as illustrated in the above figure. The input is the raw image of the current game situation. It went through several layers including convolutional layer as well as fully connected layer. The output is the Q-value for each of the actions that the agent can take.
![](Image/image.png)


Another two techniques are also essential for training DQN:
- 1. Experience Replay: Since training samples in typical RL setup are highly correlated, and less data-efficient, it will leads to harder convergence for the network. A way to solve the sample distribution problem is adopting experience replay. Essentially, the sample transitions are stored, which will then be randomly selected from the “transition pool” to update the knowledge.
- 2. Separate Target Network: The target Q Network has the same structure as the one that estimates value. Every C steps, according to the above pseudo code, the target network is reset to another one. Therefore, the fluctuation becomes less severe, resulting in more stable trainings.
### Double DQN Learning
The method of Double DQN learning is a special case of target networks. In a basic target network approach, the maximum Q-values of the target network are used. It has been seen that this approach can lead to overestimation of the Q-values, that is, the expected rewards for state/action combinations can be incorrectly large in magnitude. Instead, I implement the Double DQN approach.

The double DQN computes the Q-value of the target network using two steps:

From the target network, select the action value corresponding to the largest expected reward for the next state
Evaluate the target network at the next state using the selected action
This is a very simple modification that results in the Banana Collector solving the task in less than 300 episodes.

## Deep Q-Learning algorithm implementation details
My DQN takes as input the 37-vector of states and outputs a 4-vector of expected rewards corresponding to each action.

The DQN I used consists of a neural network with 3 dense layers. The first layer takes as input the states and has 64 nodes. The second layer takes as input the output of the first layer and has 64 nodes as well. The 3rd and final layer takes as input the output of the second layer and has 4 nodes. The first and second layers have ReLU activation functions while the final layer has a linear activation function. The DQN was implemented using [PyTorch](https://pytorch.org/).
```
- Fully connected layer - input: 37 (state size) output: 64 activation : Relu
- Fully connected layer - input: 64 output 64 activation : Relu
- Fully connected layer - input: 64 output: 4 (action size) 
```

## Target Network
Two networks were used during training, a local network and a target network. The local network is used to take actions and is updated after every time step. The target network is updated with the weights of the local network at a uniform interval. The model loss function is then the mean squared error (MSE) between the local and target network outputs. In my implementation, the target network is updated every 4 steps.

## Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 64]           2,432
            Linear-2                   [-1, 64]           4,160
            Linear-3                    [-1, 4]             260
================================================================
Total params: 6,852
Trainable params: 6,852
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.03
Estimated Total Size (MB): 0.03
----------------------------------------------------------------
```


## Hyperparameters

Q-model training parameters:
- batch size = 64  
- Optimizer : Adam
- Learning rate = 5e-4
- Loss : MSE

RL training parameters:

- BUFFER_SIZE = int(1e4)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network


## Results

### Training Plot
The agent was able to solve the environment by achieving score of 13 over 100 consecutive episodes after 274 episodes.
![Episodes vs Score Per Episode](Image/plot.jpg)

### Training Output
```
Episode 100	Average Score: 3.34	epsilon: 0.04755
Episode 200	Average Score: 8.41	epsilon: 0.00500
Episode 274	Average Score: 13.09
Environment solved in 274 episodes!	Average Score: 13.09
```

## Ideas for future work

1. A more systematic way of searching of optimal values for hyperparameters, e.g. grid search, random search, bayesian optimization or genetic algorithm
2. Prioritized Experience Replay
3. Dueling Deep Q Networks
