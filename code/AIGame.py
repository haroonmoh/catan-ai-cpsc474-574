#Settlers of Catan
#Gameplay class with pygame with AI players

from board import *
from gameView import *
from player import *
from heuristicAIPlayer import *
import queue
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt
from QlearningPlayer import QLearningPlayer, QLearningRoadAgent
 
import csv

def log_weights_csv(weights: np.ndarray, path= "weights_history.csv"):
    """
    Append a single weight vector as one row in a CSV.
    """
    weights = np.asarray(weights).flatten()

    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # optional header (w0, w1, w2, ...)
        if not file_exists:
            header = [f"w{i}" for i in range(len(weights))]
            writer.writerow(header)

        writer.writerow(weights.tolist())
        
#Class to implement an only AI
class catanAIGame():
    #Create new gameboard
    def __init__(self, road_builder_dict={}, name_dict={0: "Player1", 1: "Player2"}, num_players=2, headless=False, printless=False, eval_mode=False):
        if not printless:
            print("Initializing Settlers of Catan with only AI Players...")
        self.board = catanBoard()
        self.headless = headless
        self.printless = printless
        self.eval_mode = eval_mode
        #Game State variables
        self.gameOver = False
        self.maxPoints = 5
        self.numPlayers = num_players
        self.road_builder_dict = road_builder_dict
        self.name_dict = name_dict

        # self.numPlayers = 0

        #Dictionary to keep track of dice statistics
        self.diceStats = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        self.diceStats_list = []

        # while(self.numPlayers not in [2]): #only 2 player games allowed for now
        #     try:
        #         self.numPlayers = int(input("Enter Number of Players (2):"))
        #     except:
        #         print("Please input a valid number")
        if not printless:
            print("Initializing game with {} players...".format(self.numPlayers))
            print("Note that Player 1 goes first, Player 2 second and so forth.")
        
        #Initialize blank player queue and initial set up of roads + settlements
        self.playerQueue = queue.Queue(self.numPlayers)
        self.gameSetup = True #Boolean to take care of setup phase

        #Initialize boardview object
        self.boardView = None
        if not self.headless:
            self.boardView = catanGameView(self.board, self)

        #Functiont to go through initial set up
        self.build_initial_settlements()

        #Plot diceStats histogram
        if not self.headless:
            winner = self.playCatan()
            plt.hist(self.diceStats_list, bins = 11)
            plt.show()
        
        
        return None
    

    #Function to initialize players + build initial settlements for players
    def build_initial_settlements(self):
        #Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.numPlayers):
            
            # playerNameInput = input("Enter AI Player {} name: ".format(i+1))
            playerNameInput = self.name_dict[i]
            if i in self.road_builder_dict.keys(): # this is a Qlearning player
                newPlayer = QLearningPlayer(playerNameInput, playerColors[i], self.road_builder_dict[i], eval_mode=self.eval_mode)
                print(self.road_builder_dict)
                if not self.printless:
                    print("Player type", i, "Qlearning")
            else:
                newPlayer = heuristicAIPlayer(playerNameInput, playerColors[i])
                if not self.printless:
                    print("Player type", i, "Heuristic")
            newPlayer.updateAI()
            self.playerQueue.put(newPlayer)

        playerList = list(self.playerQueue.queue)

        #Build Settlements and roads of each player forwards
        for player_i in playerList: 
            player_i.initial_setup(self.board)
            
            if not self.headless:
                pygame.event.pump()
                self.boardView.displayGameScreen()
                pygame.time.delay(1000)


        #Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in playerList: 
            player_i.initial_setup(self.board)

            if not self.headless:
                pygame.event.pump()
                self.boardView.displayGameScreen()
                pygame.time.delay(1000)
            if not self.printless:
                print("Player {} starts with {} resources".format(player_i.name, len(player_i.setupResources)))

            #Initial resource generation
            #check each adjacent hex to latest settlement
            for adjacentHex in self.board.boardGraph[player_i.buildGraph['SETTLEMENTS'][-1]].adjacentHexList:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                if(resourceGenerated != 'DESERT'):
                    player_i.resources[resourceGenerated] += 1
                    if not self.printless:
                        print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
        if not self.headless:
            pygame.time.delay(5000)
        self.gameSetup = False


    #Function to roll dice 
    def rollDice(self):
        dice_1 = np.random.randint(1,7)
        dice_2 = np.random.randint(1,7)
        diceRoll = dice_1 + dice_2
        if not self.printless:
            print("Dice Roll = ", diceRoll, "{", dice_1, dice_2, "}")

        return diceRoll

    #Function to update resources for all players
    def update_playerResources(self, diceRoll, currentPlayer):
        if(diceRoll != 7): #Collect resources if not a 7
            #First get the hex or hexes corresponding to diceRoll
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)
            #print('Resources rolled this turn:', hexResourcesRolled)

            #Check for each player
            for player_i in list(self.playerQueue.queue):
                #Check each settlement the player has
                for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 1
                            if not self.printless:
                                print("{} collects 1 {} from Settlement".format(player_i.name, resourceGenerated))
                
                #Check each City the player has
                for cityCoord in player_i.buildGraph['CITIES']:
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacentHexList: #check each adjacent hex to a settlement
                        if(adjacentHex in hexResourcesRolled and self.board.hexTileDict[adjacentHex].robber == False): #This player gets a resource if hex is adjacent and no robber
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource.type
                            player_i.resources[resourceGenerated] += 2
                            if not self.printless:
                                print("{} collects 2 {} from City".format(player_i.name, resourceGenerated))

                if not self.printless:
                    print("Player:{}, Resources:{}, Points: {}".format(player_i.name, player_i.resources, player_i.victoryPoints))
                #print('Dev Cards:{}'.format(player_i.devCards))
                #print("RoadsLeft:{}, SettlementsLeft:{}, CitiesLeft:{}".format(player_i.roadsLeft, player_i.settlementsLeft, player_i.citiesLeft))
                if not self.printless:
                    print('MaxRoadLength:{}, Longest Road:{}\n'.format(player_i.maxRoadLength, player_i.longestRoadFlag))
        
        else:
            if not self.printless:
                print("AI using heuristic robber...")
            currentPlayer.heuristic_move_robber(self.board)


    #function to check if a player has the longest road - after building latest road
    def check_longest_road(self, player_i):
        if(player_i.maxRoadLength >= 5): #Only eligible if road length is at least 5
            longestRoad = True
            for p in list(self.playerQueue.queue):
                if(p.maxRoadLength >= player_i.maxRoadLength and p != player_i): #Check if any other players have a longer road
                    longestRoad = False
            
            if(longestRoad and player_i.longestRoadFlag == False): #if player_i takes longest road and didn't already have longest road
                #Set previous players flag to false and give player_i the longest road points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.longestRoadFlag):
                        p.longestRoadFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.longestRoadFlag = True
                player_i.victoryPoints += 2

                if not self.printless:
                    print("Player {} takes Longest Road {}".format(player_i.name, prevPlayer))

    #function to check if a player has the largest army - after playing latest knight
    def check_largest_army(self, player_i):
        if(player_i.knightsPlayed >= 3): #Only eligible if at least 3 knights are player
            largestArmy = True
            for p in list(self.playerQueue.queue):
                if(p.knightsPlayed >= player_i.knightsPlayed and p != player_i): #Check if any other players have more knights played
                    largestArmy = False
            
            if(largestArmy and player_i.largestArmyFlag == False): #if player_i takes largest army and didn't already have it
                #Set previous players flag to false and give player_i the largest points
                prevPlayer = ''
                for p in list(self.playerQueue.queue):
                    if(p.largestArmyFlag):
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prevPlayer = 'from Player ' + p.name
    
                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                if not self.printless:
                    print("Player {} takes Largest Army {}".format(player_i.name, prevPlayer))



    #Function that runs the main game loop with all players and pieces
    def playCatan(self):
        #self.board.displayBoard() #Display updated board
        numTurns = 0
        while (self.gameOver == False):
            #Loop for each player's turn -> iterate through the player queue
            for currPlayer in self.playerQueue.queue:
                numTurns += 1
                if not self.printless:
                    print("---------------------------------------------------------------------------")
                    print("Current Player:", currPlayer.name)

                turnOver = False #boolean to keep track of turn
                diceRolled = False  #Boolean for dice roll status
                
                #Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False

                while(turnOver == False):

                    #TO-DO: Add logic for AI Player to move
                    #TO-DO: Add option of AI Player playing a dev card prior to dice roll
                    
                    #Roll Dice and update player resources and dice stats
                    if not self.headless:
                        pygame.event.pump()
                    diceNum = self.rollDice()
                    diceRolled = True
                    self.update_playerResources(diceNum, currPlayer)
                    self.diceStats[diceNum] += 1
                    self.diceStats_list.append(diceNum)

                    if isinstance(currPlayer, QLearningPlayer):
                        opp = next(q for q in self.playerQueue.queue if q is not currPlayer)
                        currPlayer.move(opp, self.board)
                    else:
                        currPlayer.move(self.board) #AI Player makes all its moves
                    #Check if AI player gets longest road and update Victory points
                    self.check_longest_road(currPlayer)
                    if not self.printless:
                        print("Player:{}, Resources:{}, Points: {}".format(currPlayer.name, currPlayer.resources, currPlayer.victoryPoints))
                    if not self.headless:

                        self.boardView.displayGameScreen()#Update back to original gamescreen
                        pygame.time.delay(300)
                    turnOver = True
                    
                    #Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        self.turnOver = True
                        if not self.printless:
                            print("====================================================")
                            print("PLAYER {} WINS IN {} TURNS!".format(currPlayer.name, int(numTurns/4)))
                            print(self.diceStats)
                        else:
                            print("PLAYER {} WINS IN {} TURNS!".format(currPlayer.name, int(numTurns/4)))
                        if not self.headless:
                            print("Exiting game in 10 seconds...")
                            pygame.time.delay(10000)
                        
                        # NEW, ADDING REWARD TO QLEARNING PLAYER AND UPDATING WEIGhtS
                        for p in self.playerQueue.queue:
                            if isinstance(p, QLearningPlayer):
                                if p == currPlayer:
                                    # print("UPDATING POSITIVE REWARD")
                                    opp = next(q for q in self.playerQueue.queue if q is not p)
                                    p.road_builder.update_reward(100)
                                    p.road_builder.update_weights(0)
                                    # p.road_builder.act(p, opp, self.board, [None], eval_mode=self.eval_mode)
                                elif p != currPlayer:
                                    # print("UPDATING NEGATIVE REWARD")
                                    opp = next(q for q in self.playerQueue.queue if q is not p)
                                    p.road_builder.update_reward(-100)
                                    p.road_builder.update_weights(0)
                                    # p.road_builder.act(p, opp, self.board, [None], eval_mode=self.eval_mode)
                                
                        # break
                        return currPlayer

                if(self.gameOver):
                    if not self.headless:
                        startTime = pygame.time.get_ticks()
                        runTime = 0
                        while(runTime < 5000): #5 second delay prior to quitting
                            runTime = pygame.time.get_ticks() - startTime

                    break
    

def TrainRoadAgent(road_builder_dict, name_dict, num_players, n=1000, eval_mode=False): 
    
    for key in road_builder_dict.keys():
        log_weights_csv(road_builder_dict[key].weights)
        
    total_wins = 0
    for iter in range(n):
        
        game = catanAIGame(road_builder_dict, name_dict, num_players, headless=True, printless=True,eval_mode=eval_mode)
        winner = game.playCatan()
        
        if isinstance(winner, QLearningPlayer):
            total_wins += 1
        for key in road_builder_dict.keys():
            log_weights_csv(road_builder_dict[key].weights)
            print("WEIGHTS", road_builder_dict[key].weights)
    
    print("TOTAL WINS:", total_wins)
    print("OUT OF:", n)
    # save the weights of the road builders
    for key in road_builder_dict.keys():
        # "Saving weights to np file for loading"
        if not eval_mode:
            print("Saving weights to np file for loading")
            road_builder_dict[key].save_weights("weights/" + name_dict[key] + ".npy")

    
                          
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate Catan road agent")

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode (no learning, no exploration)"
    )

    parser.add_argument(
        "-n",
        type=int,
        default=2000,
        help="Number of games to run"
    )
    
    parser.add_argument("--weights", type=str, default="weights/QLearner.npy")

    args = parser.parse_args()

    # Initialize agent
    road_builder = QLearningRoadAgent()

    

    # Optional: load pretrained weights in eval mode
    if args.eval:
        road_builder.load_weights(args.weights)
        
    road_builder_dict = {0: road_builder}
    name_dict = {0: "QLearner", 1: "Heuristic"}

    TrainRoadAgent(
        road_builder_dict,
        name_dict,
        num_players=2,
        n=args.n,
        eval_mode=args.eval
    )
    
    # from QlearningPlayerSimple import BaseResourceFeature
    # resource = BaseResourceFeature()
    
    