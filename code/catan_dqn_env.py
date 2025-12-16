
import gym
import numpy as np
import collections
from gym import spaces
import sys
import os
import queue

# Ensure headless mode for pygame ONLY if visualization is not requested
if os.environ.get('CATAN_VISUALIZATION') != '1':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['HEADLESS'] = 'True'

from board import catanBoard, Axial_Point, Resource
from catanGame import catanGame
from player import player
from heuristicAIPlayer import heuristicAIPlayer
from hexLib import *

# -----------------------------------------------------------------------------
# 1. FIXED BOARD IMPLEMENTATION
# -----------------------------------------------------------------------------

class FixedCatanBoard(catanBoard):
    """A Catan Board with a deterministic layout for training."""
    
    def __init__(self, seed=42):
        self.seed = seed
        super().__init__()

    def getRandomResourceList(self):
        """Override to return a fixed resource list based on seed."""
        # Use a fixed deterministic random state
        rng = np.random.RandomState(self.seed)
        
        Resource_Dict = {'DESERT':1, 'ORE':3, 'BRICK':3, 'WHEAT':4, 'WOOD':4, 'SHEEP':4}
        # Fixed number list or deterministic permutation
        numbers = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
        rng.shuffle(numbers)
        
        # Flatten resources
        resources = []
        for r, count in Resource_Dict.items():
            resources.extend([r] * count)
        rng.shuffle(resources)
        
        resourceList = []
        num_idx = 0
        for r in resources:
            if r != 'DESERT':
                resourceList.append(Resource(r, numbers[num_idx]))
                num_idx += 1
            else:
                resourceList.append(Resource(r, None))
                
        return resourceList

    def updatePorts(self):
        """Override to fix port locations deterministically."""
        # We can just use the parent's method but since we control the RNG in __init__ 
        # (if we patched np.random), it might be enough. 
        # However, parent uses global np.random.permutation. 
        # We will monkey-patch global np.random temporarily or just accept randomness 
        # if the user only cared about the board tiles. 
        # For strict determinism, we should override this too.
        pass # Using default random ports for now, or we can fix them.

# -----------------------------------------------------------------------------
# 2. GYM ENVIRONMENT WRAPPER
# -----------------------------------------------------------------------------

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Monkey patch input to avoid blocking
import builtins
# Store original input just in case
_original_input = builtins.input

class HeadlessCatanGame(catanGame):
    """Subclass of catanGame that bypasses user input for setup."""
    def __init__(self, board_instance):
        print("Initializing Headless Catan Game...")
        self.board = board_instance
        
        # Game State variables
        self.gameOver = False
        self.maxPoints = 8 # Training to 8 points? Or 10? Code says 8 in one place, 10 in another.
        self.numPlayers = 2 # FORCED TO 2
        
        # Initialize blank player queue (must be queue.Queue for compatibility with existing catanGame code)
        self.playerQueue = queue.Queue(self.numPlayers)
        self.gameSetup = True 

        # Initialize boardview object (Mocked)
        self.boardView = MockView()
        
        # WE SKIP build_initial_settlements() from the constructor
        # because we want to control player creation manually in reset()
        
        return None

class CatanEnv(gym.Env):
    """
    OpenAI Gym wrapper for Catan.
    
    State Space:
    - Board Graph (Vertices, Edges)
    - Hex Tiles (Resources, Numbers)
    - Player Resources & VPs
    
    Action Space (Discrete):
    - 0: End Turn
    - 1-54: Build Settlement at Vertex i
    - 55-108: Build City at Vertex i
    - 109-180: Build Road at Edge i (approx 72 edges, need mapping)
    - 181-200: Trade Bank (4:1)
    """
    
    def __init__(self):
        self.num_players = 2
        self.seed_val = 42
        self.players = []
        # Safety: end episode after N env.step() calls even if agent never ends its turn
        self.max_steps_per_episode = 500

        # Reward shaping (optional). The env is otherwise extremely sparse-reward
        # (mostly 0 until win/loss), which can stall learning.
        #
        # Keep these small so terminal rewards (+100 / -50) remain dominant.
        self.use_intermediate_rewards = True
        self.reward_build_settlement = 1.0
        self.reward_build_city = 2.0
        self.reward_build_road = 0.2
        self.reward_trade_bank = 0.1
        # With action-masking, invalid actions should be unreachable; keep this at 0 by default.
        # If you want to debug masking failures, set this negative and/or log `info["invalid_action"]`.
        self.reward_invalid_action = 0.0
        
        # Initialize Game & Board
        self.game = None
        self.reset()
        
        # Action Space Definition
        # 0: End Turn
        # 1-54: Settlements (54 vertices)
        # 55-108: Cities (54 vertices)
        # 109-180: Roads (72 edges - we need to map them consistently)
        # 181-200: Trades (4 give -> 1 get, 5 types * 4 types = 20)
        self.action_space = spaces.Discrete(201)
        
        # Observation Space (Simplified Flattened)
        # Vertices: 54 * 3 (Owner, Type, IsPort)
        # Edges: 72 * 1 (Owner)
        # Hexes: 19 * 7 (Type OneHot + Num)
        # Players: 2 * (5 Resources + VPs + Knights + Roads)
        self.observation_space = spaces.Box(low=-1, high=100, shape=(500,), dtype=np.float32)

        # Mapping for edges
        self._map_edges()

        # Turn Stage State Machine
        # 0: Trade (Auto)
        # 1: Settlement (1 max)
        # 2: City (1 max)
        # 3: Road 1
        # 4: Road 2
        # 5: End Turn
        self.move_stage = 0

    def _map_edges(self):
        """Create a consistent index for all edges in the board graph."""
        self.edge_list = []
        visited = set()
        
        # Board graph keys are Points (pixels), not indices. 
        # But we have vertex_index_to_pixel_dict in board.
        # We prefer using vertex indices 0-53.
        
        # Sort by vertex index to ensure determinism
        sorted_vertices = range(54) 
        
        for v1_idx in sorted_vertices:
            if v1_idx not in self.game.board.vertex_index_to_pixel_dict: continue
            
            v1_pixel = self.game.board.vertex_index_to_pixel_dict[v1_idx]
            v1_obj = self.game.board.boardGraph[v1_pixel]
            
            for v2_pixel in v1_obj.edgeList:
                v2_obj = self.game.board.boardGraph[v2_pixel]
                v2_idx = v2_obj.vertexIndex
                
                # Store edge as tuple (min, max) to avoid duplicates
                edge = tuple(sorted((v1_idx, v2_idx)))
                if edge not in visited:
                    self.edge_list.append(edge)
                    visited.add(edge)
                    
        # Map edge tuple -> index
        self.edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}
        self.num_edges = len(self.edge_list)
        # print(f"Mapped {self.num_edges} unique edges.")

    def _advance_stage(self):
        """
        Advance the move_stage until we reach a stage where the agent needs to make a decision
        (i.e., a valid build action is possible) or we reach the end of the turn.
        
        Logic follows heuristicAIPlayer:
        1. Trade (Stage 0) - Automatic
        2. Build Settlement (Stage 1) - If possible
        3. Build City (Stage 2) - If possible
        4. Build Roads (Stage 3 & 4) - If possible
        """
        player = self.players[0]
        
        while self.move_stage < 5:
            # Stage 0: Auto-Trade
            if self.move_stage == 0:
                # Execute heuristic trade logic automatically
                player.trade()
                self.move_stage += 1
                continue
                
            # Stage 1: Build Settlement
            if self.move_stage == 1:
                # Check if we have resources and legal spots
                can_afford = (player.resources['BRICK'] >= 1 and player.resources['WOOD'] >= 1 and 
                              player.resources['SHEEP'] >= 1 and player.resources['WHEAT'] >= 1)
                can_place = False
                if can_afford and player.settlementsLeft > 0:
                    potentials = self.board.get_potential_settlements(player)
                    if potentials:
                        can_place = True
                
                if can_place:
                    return # STOP here, let Agent choose action
                else:
                    self.move_stage += 1 # Skip to next stage
                    continue

            # Stage 2: Build City
            if self.move_stage == 2:
                can_afford = (player.resources['WHEAT'] >= 2 and player.resources['ORE'] >= 3)
                can_place = False
                if can_afford and player.citiesLeft > 0:
                    potentials = self.board.get_potential_cities(player)
                    if potentials:
                        can_place = True
                
                if can_place:
                    return # STOP here, let Agent choose action
                else:
                    self.move_stage += 1
                    continue

            # Stage 3 & 4: Build Road
            if self.move_stage in [3, 4]:
                can_afford = (player.resources['BRICK'] >= 1 and player.resources['WOOD'] >= 1)
                can_place = False
                if can_afford and player.roadsLeft > 0:
                    potentials = self.board.get_potential_roads(player)
                    if potentials:
                        can_place = True
                
                if can_place:
                    return # STOP here
                else:
                    self.move_stage += 1
                    continue
        
        # If we reach here, self.move_stage is 5 (End Turn)
        return

    def reset(self):
        with SuppressOutput():
            # 1. Setup Board
            # We need to suppress stdout/pygame window
            self.board = FixedCatanBoard(seed=self.seed_val)
            
            # 2. Setup Game wrapper
            # Use our Headless subclass
            self.game = HeadlessCatanGame(self.board)
            self.game.numPlayers = self.num_players
            self.game.playerQueue = queue.Queue(self.num_players)
            
            # 3. Setup Players
            # Player 0: RL Agent
            # Player 1: Heuristic AI
            p1 = heuristicAIPlayer("RL_Agent", "blue")
            p2 = heuristicAIPlayer("Heuristic_AI", "red") # Use heuristic for opponent
            p1.updateAI()
            p2.updateAI()
            
            # Initial Resources (Set to standard start or empty?)
            # Standard rules: 2 settlements + 2 roads.
            # Simplification: Give them starting settlements randomly or fixed?
            # User said "Fixed initial board configuration".
            # We will use the 'initial_setup' method from heuristicAI but force it to be instant.
            
            # Store a convenient list for indexing in the environment
            self.players = [p1, p2]

            # Also populate the original game's queue structure for compatibility
            self.game.playerQueue.put(p1)
            self.game.playerQueue.put(p2)
            
            # Mock View
            self.game.boardView = MockView()
            
            # Run Initial Setup (Settlements/Roads)
            # We let them place them greedily/randomly as per their class logic
            for p in self.players:
                p.initial_setup(self.board)
                # Give starting resources based on second settlement
                if p.buildGraph['SETTLEMENTS']:
                    last_settlement = p.buildGraph['SETTLEMENTS'][-1]
                    for adj_hex in self.board.boardGraph[last_settlement].adjacentHexList:
                        r_type = self.board.hexTileDict[adj_hex].resource.type
                        if r_type != 'DESERT':
                            p.resources[r_type] += 1
            
            # Map edges again if board changed (it shouldn't)
            self._map_edges()
            
            # Start with Player 0
            self.current_player_idx = 0
            self.turn_count = 0
            self.step_count = 0
            
            # Prepare first turn state
            self.move_stage = 0
            self._advance_stage()

            return self._get_obs()

    def step(self, action):
        """
        Apply action for the current player (RL Agent).
        Then, if turn ends, play the Opponent's turn completely.
        """
        with SuppressOutput():
            player = self.players[0] # RL Agent
            reward = 0
            done = False
            info = {}

            # Count every environment interaction (independent of turns)
            self.step_count += 1
            
            # 1. Parse Action
            action_type, action_params = self._decode_action(action)
            
            # 2. Check Validity & Execute
            # Enforce Turn Order via move_stage
            valid = False
            
            # Stage 5: End Turn (Only valid action is 0)
            if self.move_stage == 5:
                if action_type == 'END_TURN':
                    valid = True
                    # Logic handled below
                else:
                    # Invalid attempt to build/trade when turn is effectively over
                    valid = False
            
            # Stage 1: Settlement
            elif self.move_stage == 1:
                if action_type == 'BUILD_SETTLEMENT':
                    v_idx = action_params
                    if self._is_valid_settlement(player, v_idx):
                        v_pixel = self.board.vertex_index_to_pixel_dict[v_idx]
                        player.build_settlement(v_pixel, self.board)
                        valid = True
                        if self.use_intermediate_rewards:
                            reward += self.reward_build_settlement
                        # Move to next stage after successful build
                        self.move_stage += 1
            
            # Stage 2: City
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
                    
            # Stage 3/4: Road
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

            # Ignore manual TRADE_BANK or out-of-order actions
            
            if not valid:
                # With strict masking, this should rarely happen unless the model predicts a masked action.
                info["invalid_action"] = True
                if self.use_intermediate_rewards:
                    reward += self.reward_invalid_action
            
            # Advance to next valid decision point
            self._advance_stage()

            # 3. Handle Turn Mechanics
            # If we reached Stage 5 and action was END_TURN, process opponent
            if valid and action_type == 'END_TURN':
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
                reward += 0 # No reward for losing
                
            if self.turn_count > 200: # Max turns (only increments on END_TURN)
                done = True
            # Safety max steps (always increments)
            if self.step_count >= self.max_steps_per_episode:
                done = True

            return self._get_obs(), reward, done, info

    def action_mask(self):
        """
        Returns a boolean mask of shape (action_space.n,) where True means the action is legal
        in the *current* state for the RL agent.
        """
        player = self.players[0]
        n = self.action_space.n
        mask = np.zeros(n, dtype=bool)

        # Always allow END_TURN only if we are at the end stage
        # mask[0] = True (REMOVED: Depends on stage)

        # Precompute legal sets once (much faster than calling per-action)
        # Note: can_settle checks resources, but we only care if we are in that stage
        # The stage advancement logic guarantees we have resources if we are in the stage.
        
        # Stage 5: End Turn
        if self.move_stage == 5:
            mask[0] = True
            return mask # Only allow end turn

        # Stage 1: Settlements
        if self.move_stage == 1:
            potential_settlements = self.board.get_potential_settlements(player)
            for v_idx, v_pixel in self.board.vertex_index_to_pixel_dict.items():
                if v_idx < 54 and v_pixel in potential_settlements:
                    mask[1 + v_idx] = True
            return mask

        # Stage 2: Cities
        if self.move_stage == 2:
            potential_cities = self.board.get_potential_cities(player)
            for v_idx, v_pixel in self.board.vertex_index_to_pixel_dict.items():
                if v_idx < 54 and v_pixel in potential_cities:
                    mask[55 + v_idx] = True
            return mask

        # Stage 3/4: Roads
        if self.move_stage in [3, 4]:
            potential_roads = self.board.get_potential_roads(player)
            for edge_idx, (v1_idx, v2_idx) in enumerate(self.edge_list):
                if v1_idx in self.board.vertex_index_to_pixel_dict and v2_idx in self.board.vertex_index_to_pixel_dict:
                    v1_p = self.board.vertex_index_to_pixel_dict[v1_idx]
                    v2_p = self.board.vertex_index_to_pixel_dict[v2_idx]
                    if (v1_p, v2_p) in potential_roads or (v2_p, v1_p) in potential_roads:
                        a = 109 + edge_idx
                        if a < n:
                            mask[a] = True
            return mask

        # Trades are automatic now, so mask remains all False (except maybe empty mask handling?)
        # Ideally we shouldn't be querying action_mask if stage is 0, but _advance_stage skips 0.
        
        return mask

    def _play_opponent_turn(self):
        """Let the heuristic AI play its turn."""
        opponent = self.players[1]
        
        # 1. Roll Dice
        roll = self.game.rollDice()
        self.game.update_playerResources(roll, opponent)
        
        # 2. Heuristic Move
        opponent.move(self.board)
        
        # 3. Check wins handled in step()

    def _roll_dice_and_distribute(self):
        roll = self.game.rollDice()
        self.game.update_playerResources(roll, self.players[0])
        # Also give to opponent? Yes, game.update_playerResources handles all players
    
    def _decode_action(self, action_idx):
        if action_idx == 0:
            return 'END_TURN', None
        elif 1 <= action_idx <= 54:
            return 'BUILD_SETTLEMENT', action_idx - 1
        elif 55 <= action_idx <= 108:
            return 'BUILD_CITY', action_idx - 55
        elif 109 <= action_idx < 109 + self.num_edges:
            return 'BUILD_ROAD', action_idx - 109
        elif 181 <= action_idx <= 200:
            # 5 resources * 4 targets = 20 trades
            # Map index to (give, get)
            res_types = ['ORE', 'BRICK', 'WHEAT', 'WOOD', 'SHEEP']
            idx = action_idx - 181
            give_i = idx // 4
            get_i = idx % 4
            if get_i >= give_i: get_i += 1 # Skip self
            return 'TRADE_BANK', (res_types[give_i], res_types[get_i])
        
        return 'UNKNOWN', None

    def _get_obs(self):
        # Construct feature vector
        obs = []
        
        # 1. Vertices (54 * 3 -> Owner, Type, IsPort)
        for i in range(54):
            if i in self.board.vertex_index_to_pixel_dict:
                v = self.board.boardGraph[self.board.vertex_index_to_pixel_dict[i]]
                owner = 0
                if v.state['Player']:
                    owner = 1 if v.state['Player'] == self.players[0] else -1
                
                type_ = 0
                if v.state['Settlement']: type_ = 1
                if v.state['City']: type_ = 2

                is_port = 1 if v.port else 0
                
                obs.extend([owner, type_, is_port])
            else:
                obs.extend([0, 0, 0])

        # 2. Edges (Roads)
        # We need to see where roads are to build networks
        for (v1_idx, v2_idx) in self.edge_list:
            v1_pixel = self.board.vertex_index_to_pixel_dict[v1_idx]
            v2_pixel = self.board.vertex_index_to_pixel_dict[v2_idx]
            v1_obj = self.board.boardGraph[v1_pixel]
            
            road_owner = 0
            # Find edge in v1's edgeList to check state
            for idx, neighbor_pixel in enumerate(v1_obj.edgeList):
                if neighbor_pixel == v2_pixel:
                    # edgeState is [Player, isRoad]
                    st = v1_obj.edgeState[idx]
                    if st[1]: # isRoad is True
                        road_owner = 1 if st[0] == self.players[0] else -1
                    break
            obs.append(road_owner)

        # 3. Hex Tiles (Resource, Number, Robber)
        # 19 hexes
        res_map = {'DESERT': 0, 'ORE': 1, 'BRICK': 2, 'WHEAT': 3, 'WOOD': 4, 'SHEEP': 5}
        for i in range(19):
            if i in self.board.hexTileDict:
                tile = self.board.hexTileDict[i]
                r_type = res_map.get(tile.resource.type, 0)
                r_num = tile.resource.num if tile.resource.num is not None else 0
                robber = 1 if tile.robber else 0
                obs.extend([r_type, r_num, robber])
            else:
                obs.extend([0, 0, 0])
                
        # 4. Player Resources
        for p in self.players:
            for r in ['ORE', 'BRICK', 'WHEAT', 'WOOD', 'SHEEP']:
                obs.append(p.resources[r])
            obs.append(p.victoryPoints)
            
        # Pad to fixed size if needed
        full_obs = np.array(obs, dtype=np.float32)
        pad = np.zeros(500 - len(full_obs))
        return np.concatenate([full_obs, pad])

    # --- Validation Helpers ---
    def _is_valid_settlement(self, player, v_idx):
        if v_idx not in self.board.vertex_index_to_pixel_dict: return False
        v_pixel = self.board.vertex_index_to_pixel_dict[v_idx]
        
        # Check resources
        if not (player.resources['BRICK']>=1 and player.resources['WOOD']>=1 and 
                player.resources['SHEEP']>=1 and player.resources['WHEAT']>=1):
            return False
            
        # Check board logic (using board.get_potential_settlements would be better but expensive to call every step)
        # We can just call get_potential_settlements and check membership
        potentials = self.board.get_potential_settlements(player)
        return v_pixel in potentials

    def _is_valid_city(self, player, v_idx):
        if v_idx not in self.board.vertex_index_to_pixel_dict: return False
        v_pixel = self.board.vertex_index_to_pixel_dict[v_idx]
        potentials = self.board.get_potential_cities(player)
        return v_pixel in potentials

    def _is_valid_road(self, player, v1_idx, v2_idx):
        if v1_idx not in self.board.vertex_index_to_pixel_dict: return False
        v1_p = self.board.vertex_index_to_pixel_dict[v1_idx]
        v2_p = self.board.vertex_index_to_pixel_dict[v2_idx]
        
        # Check edge existence implies handled by action space mapping? 
        # No, we need to check if the road is buildable (connected to our network)
        potentials = self.board.get_potential_roads(player)
        # potentials keys are (v1, v2)
        return (v1_p, v2_p) in potentials or (v2_p, v1_p) in potentials

class MockView:
    def __init__(self):
        pass
    def displayGameScreen(self):
        pass
    def displayDiceRoll(self, roll):
        pass
    def moveRobber_display(self, *args):
        return 0, None # Mock return

