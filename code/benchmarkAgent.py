import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from catanDQNEnv import CatanEnv

# -----------------------------------------------------------------------------
# 1. DQN AGENT (Same as training)
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, action_mask):
        """Pure greedy policy for benchmarking (epsilon=0)"""
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return 0

        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        mask_value = np.finfo(q.dtype).min
        q[~action_mask] = mask_value
        
        return int(np.argmax(q))

    def load(self, name):
        self.model.load_weights(name)

# -----------------------------------------------------------------------------
# 2. BENCHMARK LOGIC (Used AI to help me write this)
# -----------------------------------------------------------------------------
def run_benchmark(episodes=1000, weights_path=None):
    env = CatanEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        agent.load(weights_path)
    else:
        print("WARNING: No weights provided, running with random init (for testing)")

    wins = 0
    losses = 0
    timeouts = 0
    
    results = [] # 1=win, 0=loss/timeout
    running_win_rates = []
    
    print(f"Starting Benchmark of {episodes} games...")
    
    for e in tqdm(range(episodes)):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            mask = env.action_mask()
            action = agent.act(state, mask)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            
        # Determine outcome
        # RL Agent is Player 0. Win condition in env is VP >= 5
        # We can check env.players[0].victoryPoints
        agent_vp = env.players[0].victoryPoints
        opponent_vp = env.players[1].victoryPoints
        
        if agent_vp >= 5:
            wins += 1
            results.append(1)
        elif opponent_vp >= 5:
            losses += 1
            results.append(0)
        else:
            timeouts += 1 # Max steps reached
            results.append(0)
            
        running_win_rates.append(wins / (e + 1))

    print("\nBenchmark Completed!")
    print("--------------------------------------------------")
    print(f"Total Games: {episodes}")
    print(f"Wins: {wins} ({wins/episodes*100:.2f}%)")
    print(f"Losses: {losses} ({losses/episodes*100:.2f}%)")
    print(f"Timeouts: {timeouts} ({timeouts/episodes*100:.2f}%)")
    print("--------------------------------------------------")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(running_win_rates, label='Cumulative Win Rate')
    plt.axhline(y=wins/episodes, color='r', linestyle='--', label=f'Final Rate ({wins/episodes*100:.1f}%)')
    plt.ylim(0, 1.0)
    plt.title(f'Agent Win Rate vs Heuristic AI ({episodes} Games)')
    plt.xlabel('Games Played')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    
    output_file = 'benchmark_results.png'
    plt.savefig(output_file)
    print(f"Graph saved to {output_file}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Catan RL Agent")
    parser.add_argument("-n", "--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--weights", type=str, default="weights/catan-dqn-6360.weights.h5", help="Path to weights file")
    
    args = parser.parse_args()

    # Handle relative paths for weights
    w_path = args.weights
    if not os.path.exists(w_path):
        # Try looking in the parent directory if running from code/
        parent_path = os.path.join("..", w_path)
        if os.path.exists(parent_path):
            w_path = parent_path
        # Try looking in the current directory if just filename given
        elif os.path.exists(os.path.basename(w_path)):
             w_path = os.path.basename(w_path)
             
    run_benchmark(episodes=args.episodes, weights_path=w_path)

