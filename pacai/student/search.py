"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueueWithFunction
from pacai.util.stack import Stack


class Node:
    def __init__(self, coord, parent=None, direction=None, cost=0.0):
        self.coord = coord
        self.parent = parent
        self.direction = direction
        self.cost = cost

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


def aStarPriorityFunction(problem, heuristic):
    def priorityFunction(node):
        return node.cost + heuristic(node.coord, problem)
    return priorityFunction


def graphSearch(problem, fringe):
    fringe.push(Node(problem.startingState()))
    visited = []
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoal(node.coord):
            return node.route()
        if node.coord not in visited:
            visited.append(node.coord)
            for child in problem.successorStates(node.coord):
                fringe.push(
                    Node(child[0], node, child[1], child[2] + node.cost))
    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # <class 'pacai.core.search.position.PositionSearchProblem'>
    return graphSearch(problem, Stack())


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    return graphSearch(problem, Queue())


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    return graphSearch(problem, PriorityQueueWithFunction(priorityFunction))


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    return graphSearch(problem, PriorityQueueWithFunction
                       (aStarPriorityFunction(problem, heuristic)))
