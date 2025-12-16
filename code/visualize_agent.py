import os
import sys
import time

# Set flag to PREVENT catan_dqn_env from forcing headless mode
os.environ['CATAN_VISUALIZATION'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import gym
import pygame
import matplotlib.pyplot as plt
import queue

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from catan_dqn_env import CatanEnv, FixedCatanBoard, HeadlessCatanGame, heuristicAIPlayer
from gameView import catanGameView

# -----------------------------------------------------------------------------
# 1. DQN AGENT (Copied from Notebook)
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
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, action_mask):
        """Pure greedy policy for testing (no epsilon)"""
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            return 0

        # Predict Q-values
        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid actions
        mask_value = np.finfo(q.dtype).min
        q[~action_mask] = mask_value
        
        return int(np.argmax(q))

    def load(self, name):
        self.model.load_weights(name)

# -----------------------------------------------------------------------------
# 2. VISUAL ENVIRONMENT WRAPPER
# -----------------------------------------------------------------------------
class VisualCatanEnv(CatanEnv):
    """
    Subclass of CatanEnv that enables the GUI.
    """
    def __init__(self):
        # Force HEADLESS to False BEFORE calling super().__init__
        # But CatanEnv.__init__ calls reset() which creates the board.
        # We need to be careful about the os.environ hack in catan_dqn_env.py
        
        # We will override reset() to NOT use SuppressOutput and to create a real View.
        super().__init__()
        
    def reset(self):
        # 1. Setup Board (Fixed Seed)
        self.board = FixedCatanBoard(seed=self.seed_val)
        
        # 2. Setup Game (Use normal catanGame or modified Headless?)
        # We can use HeadlessCatanGame but we need to attach a REAL view.
        self.game = HeadlessCatanGame(self.board)
        self.game.numPlayers = self.num_players
        self.game.playerQueue = queue.Queue(self.num_players)
        
        # 3. Setup Players
        p1 = heuristicAIPlayer("RL_Agent", "blue")
        p2 = heuristicAIPlayer("Heuristic_AI", "red")
        p1.updateAI()
        p2.updateAI()
        
        self.players = [p1, p2]
        self.game.playerQueue.put(p1)
        self.game.playerQueue.put(p2)
        
        # 4. SETUP REAL VIEW
        self.game.boardView = catanGameView(self.board, self.game)
        
        # 5. Initial Setup (Instant)
        for p in self.players:
            p.initial_setup(self.board)
            if p.buildGraph['SETTLEMENTS']:
                last_settlement = p.buildGraph['SETTLEMENTS'][-1]
                for adj_hex in self.board.boardGraph[last_settlement].adjacentHexList:
                    r_type = self.board.hexTileDict[adj_hex].resource.type
                    if r_type != 'DESERT':
                        p.resources[r_type] += 1
        
        self._map_edges()
        
        self.current_player_idx = 0
        self.turn_count = 0
        self.step_count = 0
        
        # Prepare first turn
        self.move_stage = 0
        self._advance_stage()
        
        # Initial Display
        self.render()
        
        return self._get_obs()

    def _play_opponent_turn(self):
        """Override to visualize opponent actions."""
        print("--- Opponent's Turn ---")
        self.render()
        time.sleep(0.5)
        
        # We can't easily hook into the heuristic player's individual moves without modifying that class.
        # But we can call the parent logic and then render.
        super()._play_opponent_turn()
        
        self.render()
        # print("--- Opponent Finished ---")
        time.sleep(1.0)

    def step(self, action):
        # OVERRIDE step to remove SuppressOutput
        # Call parent step logic manually or copy-paste? Copy-paste is safer to ensure no suppression.
        
        player = self.players[0] # RL Agent
        reward = 0
        done = False
        info = {}

        # Count every environment interaction (independent of turns)
        self.step_count += 1
        
        # 1. Parse Action
        action_type, action_params = self._decode_action(action)
        
        # print(f"DEBUG: Processing Action {action_type} at Stage {self.move_stage}")

        # 2. Check Validity & Execute
        valid = False
        
        if self.move_stage == 5:
            if action_type == 'END_TURN':
                valid = True
            else:
                valid = False
        
        elif self.move_stage == 1:
            if action_type == 'BUILD_SETTLEMENT':
                v_idx = action_params
                if self._is_valid_settlement(player, v_idx):
                    v_pixel = self.board.vertex_index_to_pixel_dict[v_idx]
                    player.build_settlement(v_pixel, self.board)
                    valid = True
                    if self.use_intermediate_rewards:
                        reward += self.reward_build_settlement
                    self.move_stage += 1
        
        elif self.move_stage == 2:
            if action_type == 'BUILD_CITY':
                v_idx = action_params
                if self._is_valid_city(player, v_idx):
                    v_pixel = self.board.vertex_index_to_pixel_dict[v_idx]
                    player.build_city(v_pixel, self.board)
                    valid = True
                    if self.use_intermediate_rewards:
                        reward += self.reward_build_city
                    self.move_stage += 1
                
        elif self.move_stage in [3, 4]:
            if action_type == 'BUILD_ROAD':
                edge_idx = action_params
                if edge_idx < len(self.edge_list):
                    v1_idx, v2_idx = self.edge_list[edge_idx]
                    if self._is_valid_road(player, v1_idx, v2_idx):
                        v1_p = self.board.vertex_index_to_pixel_dict[v1_idx]
                        v2_p = self.board.vertex_index_to_pixel_dict[v2_idx]
                        player.build_road(v1_p, v2_p, self.board)
                        valid = True
                        if self.use_intermediate_rewards:
                            reward += self.reward_build_road
                        self.move_stage += 1

        if not valid:
            info["invalid_action"] = True
            print(f"WARNING: Invalid Action {action_type} at Stage {self.move_stage}")
            if self.use_intermediate_rewards:
                reward += self.reward_invalid_action
        
        # Advance to next valid decision point
        self._advance_stage()

        # 3. Handle Turn Mechanics
        if valid and action_type == 'END_TURN':
            # print("DEBUG: Calling Opponent Turn...")
            self._play_opponent_turn()
            self.turn_count += 1
            # Roll dice for next turn (for RL agent)
            self._roll_dice_and_distribute()
            
            # Reset for next turn
            self.move_stage = 0
            self._advance_stage()
            
        # 4. Check Win Condition
        if player.victoryPoints >= 5:
            done = True
            reward += 100
        elif self.players[1].victoryPoints >= 5:
            done = True
            reward += 0 
            
        if self.turn_count > 200:
            done = True
        if self.step_count >= self.max_steps_per_episode:
            done = True

        # Render and Sleep
        self.render()
        time.sleep(1.0)

        return self._get_obs(), reward, done, info

    def render(self):
        # Process events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        # Ensure window is visible
        if not pygame.display.get_surface():
            print("Initializing PyGame Window...")
            pygame.display.set_mode((1024, 800))
            pygame.display.set_caption("Catan AI Visualization - Agent (Blue) vs Heuristic (Red)")
            
        self.game.boardView.displayGameScreen()
        pygame.display.flip()

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Remove the dummy driver if it was set by imports
    if 'SDL_VIDEODRIVER' in os.environ:
        del os.environ['SDL_VIDEODRIVER']
    os.environ['HEADLESS'] = 'False'
    
    # Initialize Pygame
    pygame.init()
    
    # Create Env
    env = VisualCatanEnv()
    
    # Create Agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Load Weights
    weights_path = "../weights/catan-dqn-2500.weights.h5" 
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}")
        # Try absolute path based on user info if relative fails
        weights_path = "/Users/haroonmohamedali/cpsc474/catan/Catan-AI/weights/catan-dqn-2500.weights.h5"
        
    print(f"Loading weights from: {weights_path}")
    try:
        agent.load(weights_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
        
    # Run Game
    episodes = 5
    for e in range(episodes):
        print(f"Starting Episode {e+1}")
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            mask = env.action_mask()
            action = agent.act(state, mask)
            
            # Print what the agent is doing
            action_type, params = env._decode_action(action)
            print(f"Agent Action: {action_type} {params}")
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            
        print(f"Episode {e+1} finished with score: {score}")
        print("Press Enter to start next episode...")
        input()
        # time.sleep(2) # Pause between games

