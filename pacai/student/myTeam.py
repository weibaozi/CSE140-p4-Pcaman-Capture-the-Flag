from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
import random
from pacai.core.directions import Directions
from pacai.core import distance
from pacai.util.queue import Queue
from pacai.agents.capture.reflex import ReflexCaptureAgent
#modified from jguo70 serach.py
class Node:
    def __init__(self, coord, parent=None, direction=None, cost=0.0, gameState=None):
        self.coord = coord
        self.parent = parent
        self.direction = direction
        self.cost = cost
        self.gameState = gameState

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
Eacape_distance = 5
def manual_move(successorPosition,action):
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

    def __init__(self, index, isRed, treeDepth=1, **kwargs):
        super().__init__(index, **kwargs)
        self.isPacman = False
        self.isRed = isRed
        self.treeDepth = treeDepth
        self.opponents = []
        self.index = index
        print(index)

    def getTreeDepth(self):
        return self.treeDepth
    def getIndex(self):
        return self.index
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        super().registerInitialState(gameState)
        a = self.getMazeDistance((30, 14), (28, 13))
        self.opponents = self.getOpponents(gameState)
        ghost = self.getGhostPosition(gameState)
        if self.index==1:
            test_route= self.breadthFirstSearch(gameState, (19,7),index=3)
            print(test_route[1])
    # Your initialization code goes here, if you need any.
    # board = 'both' means both pacman and ghost sides, 
    # 'pacman' means only pacman, 'ghost' means only ghost
    #it will simulate the route, if the route is blocked by ghost, it will return false
    def graphSearch(self, gameState, fringe,goal, board='both',index=None):
        if index is None:
            index=self.index
        print("index", index)
        fringe.push(Node(gameState.getAgentState(index).getPosition(),gameState=gameState))
        visited = []
        while not fringe.isEmpty():
            node = fringe.pop()
            gameState=node.gameState
            if node.coord == goal:
                print("goal", goal)
                return node.route()
            if node.coord not in visited:
                visited.append(node.coord)
                actions = gameState.getLegalActions(index)
                successors = [gameState.generateSuccessor(index,action) for action in actions]
                for action, successor in zip(actions, successors):
                    if action == Directions.STOP:
                        continue
                    #print("action, successor", action, successor.getAgentState(index).getPosition())
                    if board == 'ghost': 
                        if successor.getAgentState(index).isPacman():
                            continue
                    elif board == 'pacman':
                        if not successor.getAgentState(index).isPacman():
                            continue
                    successorPosition=gameState.getAgentState(index).getPosition() 
                    successorPosition=manual_move(successorPosition,action)
                    if successorPosition == goal:
                        route= node.route() + [action]
                        return (route,len(route))
                    fringe.push(
                        Node(successor.getAgentState(index).getPosition(), 
                             node, action, 1,successor))
        return []
    def breadthFirstSearch(self,gameState, goal, board='both',index=None):
        """
        Search the shallowest nodes in the search tree first. [p 81]
        """
        return self.graphSearch(gameState, Queue(), goal, board,index)
    # unused return a list of opponent position
    def getOpponentsPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents]

    def getGhostPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents
                if not gameState.getAgentState(opponent).isPacman()]

    def getPacmanPosition(self, gameState):
        return [gameState.getAgentPosition(opponent) for opponent in self.opponents
                if gameState.getAgentState(opponent).isPacman()]
    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)
        # print("position", gameState.getAgentPosition(self.index), action)
        # for feature in features:
        #     print(feature,": {:.2f}".format(features[feature]), end=" ")
        # print("total: {:.2f}".format(stateEval))
        return stateEval
    def debug_chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        test=[(a,v) for a, v in zip(actions, values) ]
        return random.choice(bestActions)
    def chooseAction(self, gameState):
        #return self.debug_chooseAction(gameState)
        return super().chooseAction(gameState)
        # if min of opp position is less than 10 doing alpha beta
        opponentsPosition = self.getOpponentsPosition(gameState)
        if min([self.getMazeDistance(gameState.getAgentPosition(self.index), opponentPosition)
                for opponentPosition in opponentsPosition]) < 10:
            return self.max_value(gameState, 1)
        else:
            return super().chooseAction(gameState)

    def chooseAction_Pacman(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

    def chooseAction_Ghost(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

    def EvaluationFunction(self, gameState):
        # return inverse of distance of min of distance to opponent
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        print(maxValue)
        return maxValue
        # return 1/(min([self.getMazeDistance(gameState.getAgentPosition(self.index),opponentPosition) for opponentPosition in self.getOpponentsPosition(gameState)]))
        return 0

    def max_value(self, gameState, depth, alpha=-float("inf"), beta=float("inf")):
        if gameState.isWin() or gameState.isLose() or depth > self.getTreeDepth():
            return self.EvaluationFunction(gameState)
        v = -float("inf")
        best_action = Directions.STOP
        #print("index, position, actions",self.index, gameState.getAgentPosition(self.index), gameState.getLegalActions(self.index))
        result=[]
        for action in gameState.getLegalActions(self.index):
            tmp = self.min_value(gameState.generateSuccessor(
                self.index, action), depth, 0, alpha, beta)
            result.append((action,tmp))
            if tmp > v:
                v = tmp
                best_action = action
            if tmp == v:
                best_action = random.choice([best_action, action])
            # if v > beta:
            #     return v
            alpha = max(alpha, v)
        print(result)
        if depth == 1:
            return best_action
        else:
            return v

    def min_value(self, gameState, depth, agentListIndex, alpha=-float("inf"), beta=float("inf")):
        if gameState.isWin() or gameState.isLose():
            return self.EvaluationFunction(gameState)
        v = float("inf")
        agentIndex = self.opponents[agentListIndex]

        for action in gameState.getLegalActions(agentIndex):
            if agentListIndex == len(self.opponents) - 1:
                v = min(v, self.max_value(gameState.generateSuccessor(
                    agentIndex, action), depth + 1, alpha, beta))
            else:
                v = min(v, self.min_value(gameState.generateSuccessor(
                    agentIndex, action), depth, agentListIndex + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return 1


class OffenseAgent(CommonAgent):
    def __init__(self, index, isRed, **kwargs):
        super().__init__(index, isRed, **kwargs)
        self.isPacman = False
    def chooseAction(self, gameState):
        #return self.max_value(gameState, 1)
        return 'Stop'
        return super().chooseAction(gameState)

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                              for food in foodList])
            features['distanceToFood'] = minDistance
        ghostPosition = self.getGhostPosition(successor)
        pacmanPosition = self.getPacmanPosition(successor)
        # print("pacmanPosition, ghostPosition", pacmanPosition, ghostPosition)
        # if is pacman and close to a ghost, run away
        if len(ghostPosition) > 0 and successor.getAgentState(self.index).isPacman():
            minDistance = min([self.getMazeDistance(myPos, ghost)
                              for ghost in ghostPosition])
            # if minDistance <= 1:
            #     print("too close to ghost")
            #     print("ghostPosition", minDistance)
            
            features['distanceToGhost'] = 1/minDistance * 10
        # if is ghost, prefer to eat pacman
        if len(pacmanPosition) > 0 and not successor.getAgentState(self.index).isPacman():
            minDistance = min([self.getMazeDistance(myPos, pacman)
                              for pacman in pacmanPosition])
            features['distanceToPacman'] = minDistance

        if myState.isPacman():
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1
        return features
    
        

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'distanceToGhost': -1,
            'distanceToPacman': -2,
            'onDefense': -1/Eacape_distance * 10
        }


class DefenseAgent(CommonAgent):
    def __init__(self, index, isRed, **kwargs):
        super().__init__(index, isRed, **kwargs)
        self.isPacman = False
        self.initial= True

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

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
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1
        #setup initial position
        if self.initial:
            dist= self.getMazeDistance(myPos,(19,9) )
            if dist < 1:
                self.initial = False
            else:
                features['initial'] = self.getMazeDistance(myPos,(19,9) )
        ghostPosition = self.getGhostPosition(successor)
        if len(ghostPosition) > 1:
            minDistance = min([self.breadthFirstSearch(gameState, ghost,board='ghost')
                              for ghost in ghostPosition])
            features['distanceToGhost'] = 1/minDistance * 10
        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': 0,
            'initial': -0,
            'distanceToGhost': 1
        }


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

    return [firstAgent, secondAgent]
    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
