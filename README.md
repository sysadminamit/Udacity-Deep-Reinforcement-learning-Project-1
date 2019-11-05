# Udacity-Deep-Reinforcement-learning-Proje
![banana](images/banana.gif)
## Project details

As a part of **Udacity Deep Reinforcement Learning Nanodegree** course, this is the first project. In this project, an agent has to navigate and collect bananas in a large, square world. This is an assignment to understand and implement the *Deep Q-Learning method* to solve this problem.

#### Goal 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.


#### Environment Details
The state space has 37 dimensions(continous) and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is considered as episodic (actually there is no terminal state, max_time steps is choosen), and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes

```
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```


## Getting started

1. Make sure you have `python 3.6` installed.

2. Download the environment from one of the links below. You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. Install other dependencies:

```
pip install -r requirements.txt
```

## Instructions

Then open the jupyter notebook file **Navigation.ipynb** and run the cells to train the RL agent. The code is documented.
- *Replay Buffer Class Definition* : to initialize replay buffer to store experiences and random sampling
- *Agent Class Definition*: to initialize Deep Q learning agent with fixed Q targets and random experience replay. It decides the actions and also learns Q-value function.
- *Qmodel Function*: to create local and target Q-network
