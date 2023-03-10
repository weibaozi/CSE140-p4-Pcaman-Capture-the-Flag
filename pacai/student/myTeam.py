from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
import random

class TestAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, isRed, **kwargs):
        super().__init__(index, **kwargs)
        self.isPacman = False
        self.isRed = isRed

        print(index)
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """
        # currentObservation=self.getCurrentObservation()
        # agentstate=currentObservation.getAgentState(0)
        # print(agentstate.getPosition())
        if self.isPacman :
            return self.chooseAction_Pacman(gameState)
        else:
            return self.chooseAction_Ghost(gameState)

    def chooseAction_Pacman(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
    def chooseAction_Ghost(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)



def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """
    print("color",isRed)
    firstAgent = TestAgent(firstIndex,isRed)
    secondAgent = TestAgent(secondIndex,isRed)

    return [firstAgent, secondAgent]
    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
