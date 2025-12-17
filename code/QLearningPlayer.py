#Settlers of Catan
#Heuristic AI class implementation

from board import *
from player import *
import numpy as np
import random
from heuristicAIPlayer import heuristicAIPlayer

MAX_VPS = 5

#Class definition for an AI player
class QLearningPlayer(heuristicAIPlayer):
    def __init__(self, playerName, playerColor, road_builder, eval_mode=False):
        super().__init__(playerName, playerColor)
        self.road_builder = road_builder # this is a fix road_builder agent across run
        self.eval_mode = eval_mode
    
    # Only changing the "move" function to include the road-building agent
    def move(self, opp, board):
        # print("AI Player {} playing...".format(self.name))
        #Trade resources if there are excessive amounts of a particular resource
        self.trade()
        #Build a settlements, city and few roads
        
        
        
                    
        possibleVertices = board.get_potential_settlements(self)
        if(possibleVertices != {} and (self.resources['BRICK'] > 0 and self.resources['WOOD'] > 0 and self.resources['SHEEP'] > 0 and self.resources['WHEAT'] > 0)):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_settlement(list(possibleVertices.keys())[randomVertex], board) 
            # self.road_builder.update_reward(10)
        
        #Build a City
        possibleVertices = board.get_potential_cities(self)
        if(possibleVertices != {} and (self.resources['WHEAT'] >= 2 and self.resources['ORE'] >= 3)):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_city(list(possibleVertices.keys())[randomVertex], board)

        #Build a couple roads
        

        for i in range(2):
            if(self.resources['BRICK'] > 0 and self.resources['WOOD'] > 0):
                possibleRoads = board.get_potential_roads(self)
                # possibleRoads[None] = True
                road_to_build = self.road_builder.act(self, opp, board, list(possibleRoads.keys()), eval_mode=self.eval_mode)
                # randomEdge = np.random.randint(0, len(possibleRoads.keys()))
                if road_to_build == None:
                    
                    break 
                else:
                    self.build_road(road_to_build[0], road_to_build[1], board)
        # else:
            #     # possibleRoads = {None: True}
            #     # self.road_builder.act(self, opp, board, list(possibleRoads.keys()),eval_mode=self.eval_mode)
            #     # break
        
        #Draw a Dev Card with 1/3 probability
        devCardNum = np.random.randint(0, 3)
        if(devCardNum == 0):
            self.draw_devCard(board)
        
        
        
        return




class QLearningRoadAgent():
    # Q learning agent to decide where to build roads 
    
    def __init__(self):

        self.features = Features() # a list of FUNCTIONS which compute a feature value given state, action
        self.num_features = self.features.num_features
        self.weights = self.features.get_init_weights()
        # self.weights = np.ones(self.num_features)
        
        self.epsilon = 0.01
        self.alpha = 0.001
        
        # Whenever max value for next state, action is computed (i.e. vhat or qhat(s',a')), update weights corresponding to action before
        self.prev_feature_values = np.zeros(self.num_features)
        self.prev_reward = 0
        self.prev_q = 0
        
        
        # interesting statistics to track for features
        self.prev_vp_diff = 0

        # NOTE: need to make sure to do call agent to update weights one more time when the game ends as well 
        
        # IDEA: keep track of turns in between weight updates; if lots of turns have happened between a move and a second update, weigh it less
        # OR: always call, even if can't build any roads. Consider the action of "no road" to be a valid action
        # 

        
    

    
    def compute_all_q(self, player, opp, board, potential_roads):
        # returns a dict mapping from road coordinates to values
        feature_dict = {}
        q_dict = {}
        for road in potential_roads:
            feature_dict[road] = self.features.get(player, opp, board, road)
            q_dict[road] = np.dot(feature_dict[road], self.weights)
            
        return q_dict, feature_dict
    
    def update_weights(self, best_q):
        # if self.prev_reward != 0:
        #     print("UPDATING WEIGHTS WITH NONZERO REWARD")
        #     print(self.weights, self.prev_reward, best_q, self.prev_q, self.prev_feature_values)

        self.weights = self.weights + self.alpha * (self.prev_reward + best_q - self.prev_q) * self.prev_feature_values
        
        self.prev_reward = 0
        # self.prev_q = 0
        # self.prev_feature_values = None
    
    def act(self, player, opp, board, potential_roads, eval_mode=False):
        
        q_dict, feature_dict = self.compute_all_q(player, opp, board, potential_roads)

        best_road, best_q = max(q_dict.items(), key=lambda kv: kv[1])
        if not eval_mode:
            self.update_weights(best_q) # update weights for the past query using this new computed best q
        
        
        if not eval_mode:
            chosen_road = epsilon_choose(best_road, potential_roads, self.epsilon)
        else:
            chosen_road = best_road
            
        self.prev_feature_values = feature_dict[chosen_road]
        self.prev_q = q_dict[chosen_road]
        # update self.prev_reward in some way, if it is not already 
        
        return chosen_road
    
    def update_reward(self, reward):
        self.prev_reward += reward
    
    def save_weights(self, path: str):
        """
        Save weights to a .npy file.
        If `path` has no extension, we'll append '.npy'.
        """

        # ensure folder exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        np.save(path, self.weights.astype(np.float64))
        return path

    def load_weights(self, path: str):
  
        w = np.load(path)
        self.weights = w.astype(np.float64, copy=False)
 
        return self.weights
        
        
def epsilon_choose(best_choice, choices, epsilon):
    if random.random() <= epsilon:

        return random.choice(choices)
    else:
        return best_choice
        
        
        
def VP_differential_idx(player, opp):
    idx = (MAX_VPS - 2) * (player.victoryPoints - 2) + opp.victoryPoints - 2
    return idx
    
    
    
class Features():
    
    def __init__(self):
        f1 = SettlementOpportunityFeature()
        f2 = BaseResourceFeature()
        f3 = RoadBlockingFeature()
        f4 = LongestRoadFeature()
        
        # f4 = SettlementResourceFeature()
        
        self.features = [f1, f2, f3, f4] # LIST OF FEATURE-CREATING CLASSES WITH num_features and a get() function
        # Each function takes as input state, action, and outputs some k-dimensional numpy vector
        # This class should aggregate the outputs of all of these feature-creating functions into one numpy vector of total length

       
        self.num_features = 0
        for f in self.features:
            self.num_features += f.dim

        
    
    def get_init_weights(self):
        out = np.zeros(0)
        for f in self.features:
            out = np.concatenate([out, f.init_weights])
        return out
    
    def get(self, player, opp, board, road):
        out = np.zeros(0)
        for f in self.features:

            out = np.concatenate([out, f.get(player, opp, board, road)])
            
        return out
    
    
def vertex_has_player_road(player, board, v_coord, exclude_neighbor=None):

    vtx = board.boardGraph[v_coord]

    for i, nbr in enumerate(vtx.edgeList):
        if exclude_neighbor is not None and nbr == exclude_neighbor:
            continue

        owner, built = vtx.edgeState[i]
        if built and owner == player:
            return True

    return False

def vertex_has_both_player_road(player, opp, board, v_coord):

    vtx = board.boardGraph[v_coord]

    oppbool = False
    playerbool = False
    for i, nbr in enumerate(vtx.edgeList):
        owner, built = vtx.edgeState[i]
        if built and owner == player:
            playerbool = True
        if built and owner == opp:
            oppbool = True
    return (playerbool and oppbool)



def vertex_is_available(board, v_coord):
    
    vtx = board.boardGraph[v_coord]
   
    if(vtx.isColonised): #Check if this vertex is already colonised
        return False
    
    for v_neighbor in vtx.edgeList: #Check each of the neighbors from this vertex
        if(board.boardGraph[v_neighbor].isColonised):
            return False
        
    return True

    
    
class SettlementOpportunityFeature():
    # if this road opens up a new settlement spot
    def __init__(self):
        
        self.dim = (MAX_VPS - 2) ** 2
        self.init_weights = np.ones(self.dim) * 50
        # self.init_weights = np.linspace(-10, 10, num=self.dim)
        
    def get(self, player, opp, board, road):
        # return np.array([1])
        
        out = np.zeros(self.dim)
        if road == None:
            return out
        if player.victoryPoints >= MAX_VPS or opp.victoryPoints >= MAX_VPS:
            return out
        if (not vertex_has_player_road(player, board, road[0]) and vertex_is_available(board, road[0])) or (not vertex_has_player_road(player, board, road[1]) and vertex_is_available(board, road[1])):
            idx = VP_differential_idx(player, opp)
            out[idx] = 1
        return out
        
class RoadBlockingFeature():
    # If this road blocks off another player's road
    def __init__(self):
        
        self.dim = (MAX_VPS - 2) ** 2
        self.init_weights = np.ones(self.dim) * 25
        
    def get(self, player, opp, board, road):
        out = np.zeros(self.dim)
        if road == None:
            return out
        if player.victoryPoints >= MAX_VPS or opp.victoryPoints >= MAX_VPS:
            return out
        if vertex_has_both_player_road(player, opp, board, road[0]) or vertex_has_both_player_road(player, opp, board, road[1]):
            idx = VP_differential_idx(player, opp)
            out[idx] = 1
        return out
        
class BaseResourceFeature():
    # Constant feature to account for usage of resources. 
    # same across all actions, but allows us to measure against the action choice of not building a road
    def __init__(self):
        self.halfdim = (MAX_VPS - 2) ** 2 
        self.dim = self.halfdim * 2
        self.init_weights = np.ones(self.dim)
        
        base = np.zeros(self.halfdim, dtype=np.float64)

        max_diff = (MAX_VPS - 2)          # max |p - o| on this range
        for p_vp in range(2, MAX_VPS):    
            for o_vp in range(2, MAX_VPS):
                idx = (MAX_VPS - 2) * (p_vp - 2) + (o_vp - 2)

                diff = p_vp - o_vp
                val = 100.0 * diff / max_diff
                base[idx] = np.clip(val, -100.0, 100.0)
                
        self.init_weights = np.concatenate([base + 20, base], axis=0)
      
        
            
    def get(self, player, opp, board, road):

        out = np.zeros(self.dim)
        if player.victoryPoints >= MAX_VPS or opp.victoryPoints >= MAX_VPS:
            return out
        idx = VP_differential_idx(player, opp)
        if road == None:
            idx += self.halfdim
        out[idx] = 1
        return out

    
# class SettlementResourceFeature():
#     # If this road removes the player's ability to build a settlement, if they were previously able to 
#     def __init__(self):
    
#         self.dim = (MAX_VPS - 2) ** 2
#         self.init_weights = np.ones(self.dim) * -100
        
#     def get(self, player, opp, board, road):
#         out = np.zeros(self.dim)
#         if road == None:
#             return out
#         elif (player.resources['BRICK'] <= 1 or player.resources['WOOD'] <= 1) and player.resources['WHEAT'] >= 1 and player.resources['SHEEP'] >= 1:
#             idx = VP_differential_idx(player, opp)
#             out[idx] = 1
            
#         return out


# Used ChatGPT to assist with writing this helper fn
class LongestRoadFeature():
    def __init__(self):
        self.dim = (MAX_VPS - 2) ** 2
        self.init_weights = np.ones(self.dim) * 15

    def get(self, player, opp, board, road):
        out = np.zeros(self.dim)

        if road is None:
            return out
        if player.victoryPoints >= MAX_VPS or opp.victoryPoints >= MAX_VPS:
            return out

        u, v = road

        # --- compute "before" ---
        before = player.get_road_length(board) if player.buildGraph['ROADS'] else 0

        # --- snapshot board edgeState for (u,v) and (v,u) so we can restore exactly ---
        def snapshot_edge(a, b):
            vtx = board.boardGraph[a]
            for i, nbr in enumerate(vtx.edgeList):
                if nbr == b:
                    return (a, b, i, vtx.edgeState[i][0], vtx.edgeState[i][1])
            raise ValueError(f"Edge {a}->{b} not found in edgeList")

        snap_uv = snapshot_edge(u, v)
        snap_vu = snapshot_edge(v, u)

        # --- apply temporary road to board + player ---
        # add to player's buildGraph if not already present (undirected)
        added_to_player = False
        if (u, v) not in player.buildGraph['ROADS'] and (v, u) not in player.buildGraph['ROADS']:
            player.buildGraph['ROADS'].append((u, v))
            added_to_player = True

        board.updateBoardGraph_road(u, v, player)

        # --- compute "after" ---
        after = player.get_road_length(board)

        # --- revert board edgeState exactly ---
        for (a, b, i, old_owner, old_built) in (snap_uv, snap_vu):
            board.boardGraph[a].edgeState[i][0] = old_owner
            board.boardGraph[a].edgeState[i][1] = old_built

        # --- revert player buildGraph ---
        if added_to_player:
            player.buildGraph['ROADS'].pop()  # remove last appended (u,v)

        # --- feature value ---
        idx = VP_differential_idx(player, opp)

        delta = after - before
        out[idx] = max(0, delta)   # optional: only reward extensions
        # out[idx] = delta         # use this if you want negative deltas too

        return out

    
# class OverflowResourceFeature():
#     # If this road makes the player go under max hand limit (7)
#     def __init__(self):
    
#         self.dim = (MAX_VPS - 2) ** 2
#         self.init_weights = np.ones(self.dim) * 50
        
#     def get(self, player, opp, board, road):
#         out = np.zeros(self.dim)
#         if road == None:
#             return out
#         elif (player.resources['BRICK'] <= 1 or player.resources['WOOD'] <= 1) and player.resources['WHEAT'] >= 1 and player.resources['SHEEP'] >= 1
#             idx = VP_differential_idx(player, opp)
#             out[idx] = 1
            
#         return out

# class VPFeature():
#     def __init__(self):
    
#         self.dim = 1
    
#     def get(self, board, road):
        
    
        