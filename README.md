# Catan AI

This project implements AI agents for Settlers of Catan using Reinforcement Learning techniques. We compare a HeuristicAIPlayer (Greedy) vs. Q-Learning vs. DQN

## Q-Learning


## Deep Q-Network (DQN)

The DQN agent uses a neural network to approximate the Q-value function, allowing it to handle the large state space of Catan. The agent is trained against the HeuristicAIPlayer. The performance ranges from low 50s to 60 depending on how many episodes it was trained for. The difference between the HeuristicAIPlayer and the DQN agent is that instead of choosing random roads, settlements, cities to build it picks a specific location. In the end, the agent seemed to only be able to win 60% of the time compared to the greedy HeuristicAIPlayer. 

### Results Analysis

We benchmarked the DQN agent's performance at various training milestones. The following graphs show the win rate and average victory points over time.

#### Early Training (660 - 1800 Episodes)
In the early stages, the agent is still exploring the state space.

![660 Episodes](results/benchmark_results_660_episodes.png)
*Results after 660 episodes*

![1000 Episodes](results/benchmark_results_1000.png)
*Results after 1000 episodes*

![1800 Episodes](results/benchmark_results_1800_episodes.png)
*Results after 1800 episodes*

#### Mid-Training (2500 - 2940 Episodes)
As training progresses, we can observe changes in the agent's performance and strategy stability.

![2500 Episodes](results/benchmark_results_2500_episodes.png)
*Results after 2500 episodes*

![2670 Episodes](results/benchmark_results_2670_episodes.png)
*Results after 2670 episodes*

![2940 Episodes](results/benchmark_results_2940_episodes.png)
*Results after 2940 episodes*

#### Extended Training (4000 - 5310 Episodes)
With more episodes, the agent's policy should converge.

![4000 Episodes](results/benchmark_results_4000_episodes.png)
*Results after 4000 episodes*

![5310 Episodes](results/benchmark_results_5310_episodes.png)
*Results after 5310 episodes*

### Gameplay Demos

Watch the agent in action:

- [Winning Game Playthrough 1](results/Win1.mov)
- [Winning Game Playthrough 2](results/Win2.mov)

### Discussion
The progression from 660 to 5310 episodes illustrates the learning curve of the DQN agent. By comparing the graphs:
1.  **Win Rate Stability**: Notice how the win rate stabilizes or improves as the number of episodes increases.
2.  **Average Points**: The average victory points per game tend to increase, indicating the agent is learning to score more effectively even if it doesn't always win.
