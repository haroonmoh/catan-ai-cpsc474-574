# Catan AI

This project implements AI agents for Settlers of Catan using Reinforcement Learning techniques. We compare a HeuristicAIPlayer (Greedy) vs. Q-Learning vs. DQN          

We pull from an [existing Catan-AI implementation](https://github.com/kvombatkere/Catan-AI) for the rules of our game, and implement our own agents on top. We use a shorter 2-player variant of Catan with 5 VPs to win. The rest of the rules are kept the same as in the original implementation.

## Summary (in eval)
This project implements AI agents for Settlers of Catan using Reinforcement Learning techniques. We compare a HeuristicAIPlayer (Greedy) vs. Q-Learning vs. DQN    

Installation: pip install requirements.txt

Methods: 
- HeuristicAIPlayer: greedy algorithm which always builds settlements, cities, and roads when it can, randomly
- QLearner: Modification to HeuristicAIPlayer, using QLearning to decide road placements. Uses various linear approximators to estimate the value of a position/move
- DQN: Uses a neural network to approximate the Q-value function, mapping states/actions into vector formats

QLearner and DQN are both trained against HeuristicAIPlayer. 

Results: 
- QLearner vs. HeuristicAIPlayer, 52.72% (10,000 games)
- DQN vs. HeuristicAIPlayer, ~60% (1,000 games)
- DQN vs. QLearner, 79.3% (1,000 games)

Commands:
- test-all
- test-q-learning
- test-rl-agent
- test-rl-agent-long 

A full description of methods and results can be found at https://github.com/haroonmoh/catan-ai-cpsc474-574. 


## Q-Learning
The Q-Learning agent uses Q-learning with a linear approximator to choose where to place roads, while keeping everything else the same within HeuristicAIPlayer. The agent is also trained against the HeuristicAIPlayer, and is able to achieve 52.72% winrate (over 10,000 games) after training for 2,000 games. Similar to what we saw in the Q-learning for NFL pset, this is dependent on the strategy it converges to after training, and can also range from being losing (~45%) to more winning (~57%). 

### Run
To train:      
```python code/AIGame.py -n 2000  ```         
To test:         
```python code/AIGame.py -n 10000 --eval --weights weights/QLearner.npy ```


### Features
All features are indexed by both players' VP count (i.e. I have x VPs, opponent has y), each pair (x,y) gets a unique index. In other words, we only modify the corresponding index if we have the exact VP pair, otherwise it is 0. This results in a 9-dimensional feature for each. 
1. SettlementOpportunityFeature: 1 if the proposed road creates a new opportunity to build a settlement which was previously not there. 
2. BaseResourceFeature: A constant feature which is 1 if we are building a road. There is a separate index of this feature, which is 1 for the "None" action**.
3. RoadBlockingFeature: 1 if the proposed road cuts off opponent's road path 
4. LongestRoadFeature: 1 if the proposed road increases player's current longest road

**Note: after experimentation, I found that having the agent learn to build roads if it has resources available was almost always worse than just forcing it to build roads (removing the "None" feature). Thus, the second component of Feature 2 (allowing the "None" action) ended up being obsolete. However, this still gave a good baseline for the "value" of the position, simply by indexing over VP pairs, and I initialized the weights for this feature with heuristic values accordingly. 

### Rewards 
The only reward that was given was if the agent won or lost after a given move. The idea was that, in the long run, values for positions would be baked into the  q(s,a) themselves for different VP pairs; providing additional reward for gaining VPs would be like "double counting" along with the gain in value when moving from (x,y) to (x+1,y). 

### Discussion 
It was a bit challenging to figure out how to set up the reward/features, as the setup was a bit unconventional; here, the agent only acts by building roads, yet everything else that happens in between "actions" determines the success of the road that was built. I initially did not do any indexing by VPs, which led to seemingly random weights during training; I suspect this is because the model has no way to understand its current "position" in the game (i.e. winning vs. losing), and weights are only updated by which road happened to be the last one built before the game result. 

## Deep Q-Network (DQN)

The DQN agent uses a neural network to approximate the Q-value function, allowing it to handle the large state space of Catan. The agent is trained against the HeuristicAIPlayer. The performance ranges from low 50s to 60 depending on how many episodes it was trained for. The difference between the HeuristicAIPlayer and the DQN agent is that instead of choosing random roads, settlements, cities to build it picks a specific location. In the end, the agent seemed to only be able to win 60% of the time compared to the greedy HeuristicAIPlayer. 

### Results Analysis

We benchmarked the DQN agent's performance at various training milestones. The following graphs show the win rate and average victory points over time.

#### Early Training (660 - 1800 Episodes)
In the early stages, the agent quickly gets up to around 55%. 

![660 Episodes](results/benchmark_results_660_episodes.png)
*Results after 660 episodes*

![1000 Episodes](results/benchmark_results_1000.png)
*Results after 1000 episodes*

![1800 Episodes](results/benchmark_results_1800_episodes.png)
*Results after 1800 episodes*

#### Mid-Training (2500 - 2940 Episodes)
As training progresses, we can observe changes in the agent's performance and strategy stability. It gets up to 60% but hovers around 58%

![2500 Episodes](results/benchmark_results_2500_episodes.png)
*Results after 2500 episodes*

![2670 Episodes](results/benchmark_results_2670_episodes.png)
*Results after 2670 episodes*

![2940 Episodes](results/benchmark_results_2940_episodes.png)
*Results after 2940 episodes*

#### Extended Training (4000 - 5310 Episodes)
Finally, it gets to around 60% and stays there. Sometimes it dips to around 58%, but most times will get 60%

![4000 Episodes](results/benchmark_results_4000_episodes.png)
*Results after 4000 episodes*

![5310 Episodes](results/benchmark_results_5310_episodes.png)
*Results after 5310 episodes*

![6360 Episodes](results/benchmark_results_6360_episodes.png)
*Results after 6360 episodes*

### Gameplay Demos

Watch the agent in action:

- [Winning Game Playthrough 1](https://youtu.be/Afzk3vF0z_4)
- [Winning Game Playthrough 2](https://youtu.be/7l6M2oHQ1Y8)

### Discussion
Some issues I faced were hyperparameter tuning for what n and m should be, what gamma should be, and what the rewards should look like. I tried an n and m of 2 and 10, 3 and 50, 4 and 15, but ultimately 5 and 30 tended to perform the best. I also struggled with what epsilon decay should be. I had to move it up to 0.999 from 0.99 and 0.996 because the model would set on a pretty bad strategy and hover around 54% performance. For gamma and rewards, I initially had it as 0.95 and had rewards for building settlements, roads, and cities. But because of this it would just build roads and take the immediate reward. This led to it hovering around 57-58% for performance.

It was interesting to play around with DQN


## DQN vs. Q-Learning
We also played the DQN against the Q-learning agent, and the DQN agent won 79.3% of the time (over 1,000 games). This was an interested result, as this implies DQN does better against Q-learning than it does against the naive greedy agent. We hypothesize this could be that some part of Q-learning's strategy does better against HeuristicAIPlayer but is exploitable. 
