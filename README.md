# Deep-Q-Learning


## What is this?
This repository investigates the implementation of Deep Q-learning techniques within the domain of neural networks. The objective of this project is to explore Deep Q-learning in Reinforcement Learning. The model utilizes the SARSA MAX approach so the optimal Q value is chosen each time. Each episode is terminated upon the achievement of the car's position aligning with that of the designated flag.

## About the enviornment


The Mountain Car Markov Decision Process (MDP) is characterized by a deterministic environment. The car is positioned at the base of a sinusoidal valley. The car's initial placement is subject to stochasticity. In this MDP variant, the available actions are limited to accelerations that can be applied to the car in either the forward or backward direction. The primary objective within this discrete action setting is to strategically apply accelerations to guide the car to reach the top of the hill. 


There are **3** discrete actions in this enviornment. 
| Num | Observation            | Value | Unit         |
| --- | ---------------------- | ----- | ------------ |
| 0   | Accelerate to the left  | Inf   | position (m) |
| 1   | Don’t accelerate       | Inf   | position (m) |
| 2   | Accelerate to the right | Inf   | position (m) |

The observations space of the enviornment consists of shape 2. 
| Num | Observation                      | Min   | Max   | Unit         |
| --- | ---------------------------------| ----- | ----- | ------------ |
| 0   | Position of the car along the x-axis | -Inf  | Inf   | position (m) |
| 1   | Velocity of the car               | -Inf  | Inf   | position (m) |


## Approach
The Deep Q-Learning network employs the SARSA-MAX algorithm to determine optimal Q-values from the Q-Table when selecting its next state and action. In SARSA-MAX (Q-learning), the Q-value update is based on the maximum Q-value of the next state, irrespective of the action taken in that state. The Q-network is trained to approximate these Q-values, allowing the agent to make informed decisions on how to navigate the Mountain Car environment.

**SARSA MAX:** Q(s, a) ← Q(s, a) + α ⋅ [r + γ ⋅ max<sub>a'</sub> Q(s', a') - Q(s, a)]



The algorithm follows these key steps:
1. **Initialization:** Initialize the Q-network with random weights.
2. **Exploration-Exploitation:** Employ an exploration-exploitation strategy, often epsilon-greedy, to balance exploration of new actions and exploitation of known optimal actions.
3. **Action Selection:** Choose an action based on the current Q-network estimates.
4. **Environment Interaction:** Execute the selected action in the environment and observe the resulting state and reward.
5. **Q-Value Update:** Use the SARSA-MAX update rule to update the Q-value for the current state-action pair.
6. **Training:** Train the Q-network to minimize the difference between predicted and target Q-values.
7. **Iteration:** Repeat the process for multiple episodes to iteratively improve the Q-network's performance.

This approach enables the agent to learn a policy that maximizes its cumulative reward over time, ultimately guiding the car to reach the designated flag by intelligently accelerating in the Mountain Car environment.

## Problems Faced 
Initially, the agent encountered difficulty in learning how to ascend to the top of the hill due to consistently receiving a reward of -1. This posed a challenge as it provided no clear indication to the agent regarding whether it was making any incremental progress towards reaching the hill's flag at the top. 

## Solutions to Problems Faced
A solution was implementing a custom reward-shaping method. This method was introduced for the agent. The position and speed of the car were monitored and scaled by an experimentally determined coefficient. Through trial and error, this coefficient was adjusted to align with the agent's learning dynamics. The distance of the car to the top of the hill was measured, resulting in reduced penalties as the distance decreased.

Similarly, the model adapted to the car's speed, imposing lesser penalties as it increased. The agent received a positive reward of +1 upon successfully reaching the state at the top of the hill, signalling the termination of the episode.

This adaptive reward shaping method provided the agent with a more informative understanding of its environment and steps to take towards reaching the flag.

## Useful Sources
[PyTorch Doccumentation](https://pytorch.org/docs/stable/index.html) <br>
[Numpy Doccumentation](https://numpy.org/doc/)<br>
[Q-Learning Explained Simply](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial) <br>
[SARSA-MAX](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html#:~:text=The%20Sarsa%20algorithm%20is%20an,for%20updating%20the%20Q%2Dvalues.) <br>
[Open-Gym-AI Enviornments](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)

