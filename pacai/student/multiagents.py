import random
from pacai.core.directions import Directions
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best.
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        ghosts = successorGameState.getGhostPositions()
        ghostDistances_min = min(
            [distance.manhattan(newPosition, ghost) for ghost in ghosts])
        foods = oldFood.asList()
        foodDistances_min = min(
            [distance.manhattan(newPosition, food) for food in foods])
        score = successorGameState.getScore()
        result = score - 20 * foodDistances_min + 5 * ghostDistances_min
        # if ghost next to you, you will die make value -1000
        if ghostDistances_min == 1:
            result -= 1000
        # print(result)
        # print("scare:",newScaredTimes)
        return result


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):

        action = self.max_value(gameState, 1)
        # print(action)
        return action

    def max_value(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth > self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)
        v = -float("inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
            tmp = self.min_value(
                gameState.generateSuccessor(0, action), depth, 1)
            if tmp > v:
                v = tmp
                best_action = action
            if tmp == v:
                best_action = random.choice([best_action, action])
        if depth == 1:
            # print('action:',best_action,'value:',v)
            return best_action
        else:
            return v

    def min_value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(
                    gameState.generateSuccessor(agentIndex, action), depth + 1))
            else:
                v = min(v, self.min_value(gameState.generateSuccessor(
                    agentIndex, action), depth, agentIndex + 1))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):

        action = self.max_value(gameState, 1)
        # print(action)
        return action

    def max_value(self, gameState, depth, alpha=-float("inf"), beta=float("inf")):
        if gameState.isWin() or gameState.isLose() or depth > self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)
        v = -float("inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
            tmp = self.min_value(gameState.generateSuccessor(
                0, action), depth, 1, alpha, beta)
            if tmp > v:
                v = tmp
                best_action = action
            if tmp == v:
                best_action = random.choice([best_action, action])
            if v > beta:
                return v
            alpha = max(alpha, v)
        if depth == 1:
            return best_action
        else:
            return v

    def min_value(self, gameState, depth, agentIndex, alpha=-float("inf"), beta=float("inf")):
        if gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(gameState.generateSuccessor(
                    agentIndex, action), depth + 1, alpha, beta))
            else:
                v = min(v, self.min_value(gameState.generateSuccessor(
                    agentIndex, action), depth, agentIndex + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        action = self.max_value(gameState, 1)
        # print(action)
        return action

    def max_value(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth > self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)
        v = -float("inf")
        best_action = None
        for action in gameState.getLegalActions(0):
            tmp = self.min_value(
                gameState.generateSuccessor(0, action), depth, 1)
            if tmp > v:
                v = tmp
                best_action = action
            if tmp == v:
                best_action = random.choice([best_action, action])
        if depth == 1:
            # print('action:',best_action,'value:',v)
            return best_action
        else:
            return v
    # return the average value of all the actions

    def min_value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)
        v = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v += self.max_value(
                    gameState.generateSuccessor(agentIndex, action), depth + 1)
            else:
                v += self.min_value(gameState.generateSuccessor(
                    agentIndex, action), depth, agentIndex + 1)
        return v / len(gameState.getLegalActions(agentIndex))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    use score as base, add 10 for each food within 2 block,
    minus 3000 for each ghost within 2 block,
    add 30000 for each scare ghost within 2 block,
    add 10000 for each capsule within 1 block
    """
    position = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostPositions()
    foods = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.getScaredTimer()
                      for ghostState in newGhostStates]
    # print(newScaredTimes)
    result = currentGameState.getScore()
    for ghost in ghosts:
        if distance.manhattan(position, ghost) <= 2:
            if newScaredTimes[ghosts.index(ghost)] == 0:
                result -= 3000
            else:
                result += 30000
    for food in foods:
        if distance.manhattan(position, food) <= 2:
            result += 10
    for capsule in currentGameState.getCapsules():
        if distance.manhattan(position, capsule) <= 1:
            result += 10000
            # print(result)
    return result


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
