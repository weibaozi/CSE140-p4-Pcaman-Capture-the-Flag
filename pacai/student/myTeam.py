from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
import random
from pacai.core.directions import Directions
from pacai.core import distance
from pacai.util.queue import Queue
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.search.position import PositionSearchProblem
from pacai.util.priorityQueue import PriorityQueueWithFunction
from pacai.core import distanceCalculator
from pacai.agents.capture.defense import DefensiveReflexAgent
import time
ChaseDistance = 4  # chase distance
OnDefenseMultiplier = 30
SafeDistToAttack = 3  # keep this distance to ghost when trying to attack
SafeDistToScaredGhost = 3  # twice this distance is the lefter timer for scared ghost
SafeDistToDefend = 2  # keep this distance to pacman when get scared
TrapSpace = 25
# modified from jguo70 serach.py


class Node:
    def __init__(self, coord, parent=None, direction=None, cost=0.0, gameState=None):
        self.coord = coord
        self.parent = parent
        self.direction = direction
        self.cost = cost
        self.gameState = gameState
        self.find = True
        self.visited = []

    def __repr__(self):
        return f'Node:{self.coord }, parent:{self.parent.coord if self.parent else None},\
              direction:{self.direction}, priority:{self.cost}'

    def __lt__(self, other):
        return self.cost < other.cost

    def route(self):
        if self.parent:
            return self.parent.route() + [self.direction]
        else:
            return []


def priorityFunction(node):
    return node.cost
# ranges and opponents and costrains, length is the max length of the route

# graph search, have some constrains such asa range and opponents position


def graphSearch(problem, fringe, ranges=(0, 100), opponents=[], hard=False):
    fringe.push(Node(problem.startingState()))
    visited = []
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoal(node.coord):
            return node
        if node.coord not in visited:
            visited.append(node.coord)
            for child in problem.successorStates(node.coord):
                cost = 1
                # if child[0] in opponents' position ,skip
                if child[0] in opponents:
                    if hard:
                        continue
                    cost += 100
                # makes cost high if goes out of range

                if child[0][0] not in ranges:
                    if hard:
                        continue
                    cost += 100
                fringe.push(
                    Node(child[0], node, child[1], cost + node.cost))
    # print(visited)
    n = Node(problem.startingState(), cost=10000)
    n.visited = visited
    n.find = False
    return n

# def priorityFunction(node):
#     return node.cost

# def aStarPriorityFunction(problem, heuristic):
#     def priorityFunction(node):
#         return node.cost + heuristic(node.coord, problem)
#     return priorityFunction


# def aStarSearch(problem, heuristic):
#     """
#     Search the node that has the lowest combined cost and heuristic first.
#     """
#     # *** Your Code Here ***
#     return graphSearch(problem, PriorityQueueWithFunction
#                        (aStarPriorityFunction(problem, heuristic)))

def manual_move(successorPosition, action):
    if action == Directions.NORTH:
        successorPosition = (successorPosition[0], successorPosition[1] + 1)
    elif action == Directions.SOUTH:
        successorPosition = (successorPosition[0], successorPosition[1] - 1)
    elif action == Directions.EAST:
        successorPosition = (successorPosition[0] + 1, successorPosition[1])
    elif action == Directions.WEST:
        successorPosition = (successorPosition[0] - 1, successorPosition[1])
    return successorPosition


class CommonAgent(ReflexCaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, isRed, **kwargs):
        super().__init__(index, **kwargs)
        self.isPacman = False
        self.isRed = isRed
        self.opponents = []
        self.index = index
        self.height = 0
        self.width = 0
        self.initialLocation = None
        self.state = None
        self.walls = None
        self.stopTimes = 0
        self.friendIndex = None
        self.lastInvader = None
        self.trappedPacman = None
        self.debug = False
        self.debugHard = False
        self.debugPosition = None
        print(index)
    # def advancedMazeDistance(self, gameState, start, end, side='both',opponents=[]):
    #     walls=gameState.getWalls()
    #     for opponent in opponents:
    #         walls[opponent[0]][opponent[1]]=True

    def printd(self, *args):
        if self.debug:
            print(*args)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        self.state = "start"
        super().registerInitialState(gameState)
        self.height, self.width = gameState.getWalls(
        ).getHeight(), gameState.getWalls().getWidth()

        self.opponents = self.getOpponents(gameState)

        self.initialLocation = gameState.getAgentState(
            self.index).getPosition()
        print("index, gride size, initial location", self.index,
              self.height, self.width, self.initialLocation)
        self.friendIndex = self.getTeam(gameState).copy()
        self.friendIndex.remove(self.index)
        self.friendIndex = self.friendIndex[0]

        self.walls = gameState.getWalls()

    def debug_chooseAction(self, gameState):
        # goto debug position
        actions = gameState.getLegalActions(self.index)
        nextStates = [gameState.generateSuccessor(self.index, action)
                      for action in actions]
        values = [self.getMazeDistance(state.getAgentPosition(self.index), self.debugPosition)
                  for state in nextStates]
        best = min(values)
        bestActions = [a for a, v in zip(actions, values) if v == best]
        return random.choice(bestActions)

    def chooseAction(self, gameState):
        # return self.debug_chooseAction(gameState)
        if self.debugHard:
            if self.debugPosition is not None:
                return self.debug_chooseAction(gameState)
            else:
                return Directions.STOP

        return super().chooseAction(gameState)

    def ghostScared(self, gameState):
        numAgents = gameState.getNumAgents()
        return [gameState.getAgentState(i).getScaredTimer() > SafeDistToScaredGhost*2 for i in range(numAgents)]

    def getRange(self, isRed, side='both'):
        # get range of the game board:
        half = self.width//2
        if side == 'both':
            return range(1, self.width-1)
        elif side == 'pacman':
            if isRed:
                return range(half, self.width-1)

            else:
                return range(1, half)
        elif side == 'ghost':
            if isRed:
                return range(1, half)
            else:
                return range(half, self.width-1)

        else:
            raise Exception('side should be one of [both, pacman, ghost]')

    def getIndex(self):
        return self.index

    # board = 'both' means both pacman and ghost sides,
    # 'pacman' means only pacman, 'ghost' means only ghost
    # it will simulate the route, if the route is blocked by ghost, it will return false
    def UCS(self, gameState, goal, start=None, side='both', index=None, opponents=[], hard=False):
        if index is None:
            index = self.index
        if start is None:
            start = gameState.getAgentState(index).getPosition()
        problem = PositionSearchProblem(gameState, goal=goal, start=start)
        ranges = self.getRange(
            (not self.isRed) if index in self.opponents else self.isRed, side)
        return graphSearch(problem, PriorityQueueWithFunction(priorityFunction), ranges, opponents, hard=hard)

    # unused return a list of opponent position
    def getOpponentsPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents]

    def getNonScaredGhostPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents
                if not gameState.getAgentState(opponent).getScaredTimer() > SafeDistToScaredGhost*2
                and not gameState.getAgentState(opponent).isPacman()]

    def getScaredGhostPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents
                if gameState.getAgentState(opponent).getScaredTimer() > SafeDistToScaredGhost*2
                and not gameState.getAgentState(opponent).isPacman()]

    def getGhostPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents
                if not gameState.getAgentState(opponent).isPacman()]

    def getPacmanPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents
                if gameState.getAgentState(opponent).isPacman()]

    def getGhostIndex(self, gameState):
        return [opponent for opponent in self.opponents
                if not gameState.getAgentState(opponent).isPacman()]

    def getPacmanIndex(self, gameState):
        return [opponent for opponent in self.opponents
                if gameState.getAgentState(opponent).isPacman()]

    def isPacmanTrapped(self, gameState, pacmanIndex, ghostIndexes):
        if pacmanIndex == None or ghostIndexes == None:
            return False
        pacmanPosition = gameState.getAgentPosition(pacmanIndex)
        pacmanActions = gameState.getLegalActions(pacmanIndex)
        nextPacmanPositions = [gameState.generateSuccessor(pacmanIndex, action).getAgentPosition(pacmanIndex)
                               for action in pacmanActions]
        for ghost in ghostIndexes:
            nextGhostPosition = gameState.getAgentPosition(ghost)
            legalActions = gameState.getLegalActions(ghost)
            nextGhostPositions = [gameState.generateSuccessor(ghost, action).getAgentPosition(ghost)
                                  for action in legalActions]
            # explore routes to see if there is a way to escape
            result = self.UCS(gameState, start=pacmanPosition, goal=None, opponents=[
                              nextGhostPosition], hard=True)
            num = 0
            for p in nextGhostPositions:
                if p in result.visited:
                    num += 1
            if num == 1 and len(result.visited) < TrapSpace:
                self.printd("pacman trapped")
                return True
            PacmanMoves = [
                p in nextGhostPositions for p in nextPacmanPositions]
            if all(PacmanMoves):
                self.printd("all in my action", PacmanMoves)
                return True
        return False

    def getSurround(self, position):
        x, y = position
        tmp = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        result = []
        for p in tmp:
            if not self.walls[p[0]][p[1]]:
                result.append(p)
        return result

    def getEatableFood(self, gameState, ghostPosition):
        foods = self.getFood(gameState).asList()
        result = []
        for food in foods:
            cost = self.UCS(gameState, food, opponents=self.getNonScaredGhostPosition(
                gameState), hard=True).cost
            foodSurround = self.getSurround(food)
            maxDistFoodToGhost = max([self.getMazeDistance(
                foodp, ghostPosition) for foodp in foodSurround])
            if maxDistFoodToGhost - cost >= SafeDistToDefend:
                result.append(food)
        return result

    def EvaluationFunction(self, gameState):
        # return inverse of distance of min of distance to opponent

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        # print([(a,v) for a,v in zip(actions,values)])
        return maxValue
        # return 1/(min([self.getMazeDistance(gameState.getAgentPosition(self.index),opponentPosition) for opponentPosition in self.getOpponentsPosition(gameState)]))
        return 0


class OffenseAgent(CommonAgent):
    def __init__(self, index, isRed, **kwargs):
        super().__init__(index, isRed, **kwargs)
        self.isPacman = False

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        if self.index == 0:
            self.debugPosition = (23, 8)
            self.debug = False

    def chooseAction(self, gameState):

        ghostPosition = self.getNonScaredGhostPosition(gameState)
        capsulePosition = self.getCapsules(gameState)
        ghostMinDistance = 9999
        myPos = gameState.getAgentPosition(self.index)

        isPacman = gameState.getAgentState(self.index).isPacman()
        self.isPacman = isPacman
        if len(ghostPosition) > 0:
            ghostMinDistance = min([self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), ghost)
                                    for ghost in ghostPosition])
        if gameState.getAgentState(self.index).isPacman():
            self.isPacman = True
        else:
            self.isPacman = False

        # State Machine
        # start: goto defense
        if self.state == "start":
            self.state = 'defense'
        # defense: goto attack if no ghost
        elif self.state == "defense":
            if isPacman:
                self.state = "attack"
                if ghostMinDistance <= ChaseDistance:
                    self.state = "chase"
        # attack: goto chase if ghost is near and goto power if ghost is scared
        elif self.state == "attack":
            if len(self.getScaredGhostPosition(gameState)) > 0:
                self.state = "power"
            if ghostMinDistance <= ChaseDistance:
                self.state = "chase"
        # chase: goto power if ghost is scared and goto defense if no pacman
        elif self.state == "chase":
            if len(self.getScaredGhostPosition(gameState)) > 0:
                self.state = "power"
            if not isPacman:
                self.state = "defense"
            elif ghostMinDistance > ChaseDistance:
                self.state = "attack"

        # power: after all ghost is not scared goto chase if ghost is near or goto attack
        elif self.state == "power":
            if len(self.getScaredGhostPosition(gameState)) == 0:
                if ghostMinDistance <= ChaseDistance:
                    self.state = "chase"
                else:
                    self.state = "attack"
        else:
            raise Exception("state error")
        if self.debug:
            print("")
            print(myPos, self.state)

        action = super().chooseAction(gameState)
        if action == Directions.STOP:
            self.stopTimes += 1
        return action

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature]
                        for feature in features)
        if self.debug:
            print(action)
            for feature in features:
                print(feature, ": {:.2f}".format(features[feature]), end=" ")
            print("")
            print("total: {:.6f}".format(stateEval))
        return stateEval

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        oldPos = gameState.getAgentState(self.index).getPosition()
        myPos = myState.getPosition()
        isPacman = myState.isPacman()
        ghostPosition = self.getNonScaredGhostPosition(successor)
        pacmanPosition = self.getPacmanPosition(successor)
        scaredGhostPosition = self.getScaredGhostPosition(successor)
        friendPos = gameState.getAgentPosition(self.friendIndex)
        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()
        capsulePosition = self.getCapsules(successor)
        numsOfScaredGhost = len(scaredGhostPosition)

        features['successorScore'] = self.getScore(successor)
        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                              for food in foodList])
            # minDistance = min([self.UCS(successor, food,myPos,opponents=ghostPosition).cost
            #                   for food in foodList])
            # print("minDistance", minDistance)
            features['distanceToFood'] = minDistance

        # distance to ghost
        if len(ghostPosition) > 0:
            minDistance = min([self.getMazeDistance(myPos, ghost)
                              for ghost in ghostPosition])
            features['distanceToGhost'] = minDistance
            if minDistance == 1:
                features['onDying'] = 1
            if minDistance <= ChaseDistance:
                features['onChase'] = 1
            features['distanceToGhostInverse'] = 1.0 / minDistance

        # distance to capsule
        if len(capsulePosition) > 0:
            distance = self.UCS(
                successor, capsulePosition[0], myPos, opponents=ghostPosition).cost
            features['distanceToCapsule'] = distance
        features['numCapsules'] = len(capsulePosition)
        # distance to scared ghost
        if numsOfScaredGhost > 0:
            minDistance = len([self.getMazeDistance(myPos, ghost)
                              for ghost in scaredGhostPosition])
            features['distanceToScaredGhost'] = minDistance

        # Pacman distance
        if len(pacmanPosition) > 0 and not isPacman:
            minDistance = min([self.getMazeDistance(myPos, pacman)
                              for pacman in pacmanPosition])
            features['distanceToPacman'] = minDistance
        if friendPos != None:
            features['distanceToFriend'] = self.getMazeDistance(
                myPos, friendPos)

        # if next local is respawn point and not by moving, then it is respawn
        if myPos == self.initialLocation and self.getMazeDistance(oldPos, self.initialLocation) > 1:
            features['RespawnLocation'] = 1
        # Stop and reverse check
        if (action == Directions.STOP):
            features['stop'] = 1+self.stopTimes
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1
        # on which side of the map
        if myState.isPacman():
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1

        if self.state == "chase":
            homeDistance = self.UCS(
                successor, self.initialLocation, opponents=ghostPosition).cost
            features['distanceToEscape'] = homeDistance
            if len(ghostPosition) > 0:
                eatableFood = self.getEatableFood(gameState,
                                                  min(ghostPosition, key=lambda x: self.getMazeDistance(myPos, x)))
                if len(eatableFood) > 0:
                    #print("eatableFood", eatableFood)
                    minDistance = min([self.getMazeDistance(myPos, food)
                                       for food in eatableFood])
                    features['distanceToEatableFood'] = minDistance
        elif self.state == "power":
            features['numOfScaredGhost'] = numsOfScaredGhost
        elif self.state == "defense":
            pass
            if features['onDefense'] == 0:
                features['distanceToGhostInverse'] *= 2

        return features

    def getWeights(self, gameState, action):
        offenseCommon = {
            'successorScore': 100,
            'distanceToFood': -3,
            'distanceToGhost': 1,
            'distanceToPacman': -1,
            'onDefense': 0,
            'distanceToCapsule': 0,
            'distanceToEscape': -6,
            'distanceToScaredGhost': -1,
            'RespawnLocation': -1000000,
            'numOfScaredGhost': -40,
            'stop': -5,
            'reverse': 0,
            'numCapsules': -100,
            'onDying': -1000000,
            'distanceToGhostInverse': 0,
            'onChase': 0,
            'distanceToFriend': 0,
            'distanceToEatableFood': 0
        }
        stateWeight = {}
        if self.state == "defense":
            stateWeight = {
                'onDying': 0,
                'reverse': -5,
                'Stop': -5,
                'distanceToFood': -3,
                'distanceToGhost': 1,
                'distanceToPacman': -1,
                'onDefense': -OnDefenseMultiplier,
                'distanceToCapsule': -5,
                'distanceToGhostInverse': -SafeDistToAttack*OnDefenseMultiplier,
                'distanceToFriend': 2,
            }
        elif self.state == "attack":
            stateWeight = {

                'distanceToCapsule': -0.5,
                'numCapsules': 0
            }
        elif self.state == "power":
            stateWeight = {
                'distanceToCapsule': -1,
                'distanceToGhost': 1,
                'numCapsules': -300,
                'numOfScaredGhost': -40,
                'distanceToScaredGhost': -1,
            }
        elif self.state == "chase":
            stateWeight = {
                'successorScore': 30,
                'distanceToFood': -0.5,
                'distanceToCapsule': -20,
                'numCapsules': -1000,
                'distanceToEscape': -1,
                'distanceToGhost': -3,
            }
        offenseCommon.update(stateWeight)
        return offenseCommon


class DefenseAgent(CommonAgent):
    def __init__(self, index, isRed, **kwargs):
        super().__init__(index, isRed, **kwargs)
        self.isPacman = False
        self.state = "start"

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        if self.index == 3:
            self.debug = True
            self.debugPosition = None

    def chooseAction(self, gameState):
        scareTime = gameState.getAgentState(self.index).getScaredTimer()
        pacmanPosition = self.getPacmanPosition(gameState)
        myPos = gameState.getAgentPosition(self.index)
        invaders = self.getPacmanIndex(gameState)
        minDistInvader = None
        if len(invaders) > 0:
            minDistInvader = min(invaders, key=lambda x: self.getMazeDistance(myPos,
                                                                              gameState.getAgentPosition(x)))
            self.lastInvader = minDistInvader
            self.friendAgent.lastInvader = minDistInvader
        if self.state == "start":
            self.state = "target"
        elif self.state == "target":
            if len(pacmanPosition) > 0:
                self.state = "chase"
        elif self.state == "chase":
            if len(pacmanPosition) == 0:
                self.state = "target"
            elif self.isPacmanTrapped(gameState, minDistInvader, [self.index]):
                self.trappedPacman = minDistInvader
                self.state = "trap"
            if scareTime > 0:
                self.state = "scared"
        elif self.state == "scared":
            if scareTime == 0:
                if len(pacmanPosition) > 0:
                    self.state = "chase"
                else:
                    self.state = "target"
        elif self.state == "trap":
            if scareTime > 0:
                self.state = "scared"
            else:
                if not self.isPacmanTrapped(gameState, self.trappedPacman, [self.index]):
                    if len(pacmanPosition) > 0:
                        self.state = "chase"
                    else:
                        self.state = "target"
        else:
            raise Exception("Unknown state")
        if self.debug:
            print("")
            print(myPos, self.state, self.friendAgent.state,
                  self.friendAgent.lastInvader)
        return super().chooseAction(gameState)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature]
                        for feature in features)
        if self.debug:
            print(action)
            for feature in features:
                print(feature, ": {:.2f}".format(features[feature]), end=" ")
            print("")
            print("total: {:.6f}".format(stateEval))
        return stateEval

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        oldPos = gameState.getAgentState(self.index).getPosition()
        myPos = myState.getPosition()
        ghostPosition = self.getGhostPosition(successor)
        foodList = self.getFoodYouAreDefending(successor).asList()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0
        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman(
        ) and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.UCS(successor, a.getPosition(),
                              side='ghost').cost for a in invaders]
            features['invaderDistance'] = min(dists)
            features['safeDistanceToInvader'] = abs(
                min(dists) - SafeDistToDefend)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1
        # ghost distance
        if len(ghostPosition) > 0:
            minDistGhost = min(self.getGhostIndex(successor), key=lambda x: self.getMazeDistance(
                myPos, successor.getAgentPosition(x)))
            if self.lastInvader is not None:
                minDistGhost = self.lastInvader
            minDistance = self.UCS(successor, successor.getAgentPosition(minDistGhost),
                                   side='ghost').cost
            features['distanceToGhost'] = minDistance
            features['safeDistanceToGhost'] = abs(
                minDistance - SafeDistToDefend)

        if (len(foodList) > 0):
            minDistance = min([self.UCS(successor, food, myPos, opponents=ghostPosition).cost
                              for food in foodList])
            features['distanceToMyFood'] = minDistance

        # if next local is respawn point and not by moving, then it is respawn
        if myPos == self.initialLocation and self.getMazeDistance(oldPos, self.initialLocation) > 1:
            features['RespawnLocation'] = 1
        return features

    def getWeights(self, gameState, action):
        offenseCommon = {
            'numInvaders': -1000,
            'onDefense': 100000,
            'invaderDistance': -10,
            'stop': -10,
            'reverse': -2,
            'initial': -0,
            'distanceToGhost': 0,
            'RespawnLocation': -1000000,
            'safeDistanceToGhost': 0,
            'safeDistanceToInvader': 0,
            'distanceToMyFood': 0
        }
        stateWeight = {}
        if self.state == "target":
            stateWeight = {
                'distanceToGhost': 0,
                'safeDistanceToGhost': -3,
                'reverse': 0,
            }
        elif self.state == "chase":
            stateWeight = {

            }
        elif self.state == "scared":
            stateWeight = {
                'invaderDistance': 0,
                'safeDistanceToInvader': -1,
                'stop': 0,
                'reverse': 0,
            }
        elif self.state == "trap":
            stateWeight = {
                'stop': 100000000,
            }
        offenseCommon.update(stateWeight)
        return offenseCommon


def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.agents.capture.dummy.DummyAgent',
               second='pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """
    print("color", isRed)
    firstAgent = OffenseAgent(firstIndex, isRed)
    secondAgent = DefenseAgent(secondIndex, isRed)
    firstAgent.friendAgent = secondAgent
    secondAgent.friendAgent = firstAgent
    #secondAgent = DefensiveReflexAgent(secondIndex)

    return [firstAgent, secondAgent]
    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
