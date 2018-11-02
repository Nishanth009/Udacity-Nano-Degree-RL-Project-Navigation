# Udacity-Nano-Degree-RL-Project-Navigation

![](/images/navigation.gif)

## Project Details

This project was done as part of deep reinforcement learning nanodegree by Udacity. Here we train an AI agent to navigate through a large, square world and collect bananas.A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. A trained agent will collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.The problem is episodic and agent must get an average score of +13 over 100 consecutive episodes to solve the environment 

Four actions are,

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

## Getting Started


Install anaconda(python package manager)-python 3, then run ./install.sh file.This will install unity ml-agents, torch and other neccessery libraries


## Instructions

Run **Navigation.py**. The agent uses Deep Q-Network algorithm to learn appropriate action for a state. The input to the deep neural network is the state vector and the output will be an action. DQN uses Experience Replay and Target network to get improved performace.The agents performance is measured based on its mean score over 100 episodes.In this case agent should get an average score of +13 over 100 consecutive episodes to solve the environment.Please check report.md for more details.
