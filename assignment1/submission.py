## ID: 20160169 NAME: Choi Soomin
######################################################################################
# Problem 2a
# minimax value of the root node: 6
# pruned edges: h, m, x
######################################################################################

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)

    numAgents = gameState.getNumAgents()

    def inspectDepth(depth, currentState, agentIndex):
      if depth == self.depth or currentState.isWin() or currentState.isLose():
        return (self.evaluationFunction(currentState), Directions.STOP)

      bestScore = float('-inf') if agentIndex == 0 else float('inf')
      bestMove = Directions.STOP

      legalMoves = currentState.getLegalActions(agentIndex)
      # print(depth, agentIndex, legalMoves)
      for legalMove in legalMoves:
        nextState = currentState.generateSuccessor(agentIndex, legalMove)
        (score, move) = inspectDepth(depth + (agentIndex + 1) // numAgents, nextState, (agentIndex + 1) % numAgents)

        if agentIndex == 0:
          # print("\ttrial!", agentIndex, score, bestScore)
          if score > bestScore:
            bestScore = score
            bestMove = legalMove
        elif score < bestScore:
          # print("\ttrial 2!", agentIndex, score, bestScore)
          bestScore = score
          bestMove = bestMove

      return (bestScore, bestMove)

    nextScore, nextMove = inspectDepth(0, gameState, 0)

    # print(nextScore, nextMove, "is the max score!")

    return nextMove

    # END_YOUR_ANSWER

######################################################################################
# Problem 2b: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER (our solution is 42 lines of code, but don't worry if you deviate from this)

    numAgents = gameState.getNumAgents()

    def inspectDepth(depth, currentState, agentIndex, alpha, beta):
      if depth == self.depth or currentState.isWin() or currentState.isLose():
        return (self.evaluationFunction(currentState), Directions.STOP)

      bestScore = alpha if agentIndex == 0 else beta
      bestMove = Directions.STOP

      legalMoves = currentState.getLegalActions(agentIndex)
      # print(depth, agentIndex, legalMoves)
      for legalMove in legalMoves:
        nextState = currentState.generateSuccessor(agentIndex, legalMove)
        (score, move) = inspectDepth(depth + (agentIndex + 1) // numAgents, nextState, (agentIndex + 1) % numAgents, alpha, beta)

        if agentIndex == 0:
          # print("\ttrial!", agentIndex, score, bestScore)
          if score > bestScore:
            bestScore = score
            alpha = score
            bestMove = legalMove
          if score >= beta: break
        else:
          # print("\ttrial 2!", agentIndex, score, bestScore)
          if score < bestScore:
            bestScore = score
            beta = score
            bestMove = bestMove
          if score <= alpha: break

      return (bestScore, bestMove)

    nextScore, nextMove = inspectDepth(0, gameState, 0, float('-inf'), float('inf'))
    # print(nextScore, nextMove, "is the max score!")

    return nextMove

    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)

    numAgents = gameState.getNumAgents()

    def inspectDepth(depth, currentState, agentIndex):
      if depth == self.depth or currentState.isWin() or currentState.isLose():
        return (self.evaluationFunction(currentState), Directions.STOP)

      bestScore = float('-inf') if agentIndex == 0 else float('inf')
      expScore = 0
      bestMove = Directions.STOP

      legalMoves = currentState.getLegalActions(agentIndex)
      # print(depth, agentIndex, legalMoves)
      for legalMove in legalMoves:
        nextState = currentState.generateSuccessor(agentIndex, legalMove)
        (score, move) = inspectDepth(depth + (agentIndex + 1) // numAgents, nextState, (agentIndex + 1) % numAgents)

        if agentIndex == 0:
          # print("\ttrial!", agentIndex, score, bestScore)
          if score > bestScore:
            bestScore = score
            bestMove = legalMove
        else:
          expScore += score / len(legalMoves)
          # print("\ttrial 2!", depth, legalMove, agentIndex, score, expScore, len(legalMoves))

      return (bestScore, bestMove) if agentIndex == 0 else (expScore, bestMove)

    nextScore, nextMove = inspectDepth(0, gameState, 0)

    # print(nextScore, nextMove, "is the max score!")

    return nextMove

    # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 4).
  """

  # BEGIN_YOUR_ANSWER (our solution is 60 lines of code, but don't worry if you deviate from this)

  # distance from ghost - far is better.
  # food count - less is better. reciprocal.
  # distance from food - less is better. reciprocal
  # capsule - less is better. reciprocal

  wall = currentGameState.getWalls()
  maxDistance = wall.width + wall.height - 4
  pacmanPos = currentGameState.getPacmanPosition()
  numAgents = currentGameState.getNumAgents()

  def getDistances(positions):
      return [manhattanDistance(pacmanPos, pos) for pos in positions]

  def ghostFeature():
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    distances = getDistances(currentGameState.getGhostPositions())

    minScaredDistance = float('inf')
    minGhostDistance = float('-inf')
    additional = 0
    countScared = 0
    for i, scaredTime in enumerate(scaredTimes):
      if scaredTime:
        # if distances[i] <= 5:
        additional += 300 / (distances[i] + 1)
        if minScaredDistance > distances[i]:
          minScaredDistance = distances[i]
        countScared += 1
      else:
        if distances[i] < 2:
          additional -= 300 / (distances[i] + 1)
        if minGhostDistance > distances[i]:
          minGhostDistance = distances[i]
    return additional + (numAgents - countScared) * 100 + 10 / minScaredDistance - 10 / minGhostDistance

  def foodFeature():
    foods = currentGameState.getFood().asList()
    distances = getDistances(foods)
    minDistance = min(distances) if len(distances) > 0 else maxDistance
    return 180 / (sum(distances) + 1) - len(foods) * 10

  def capsuleFeature():
    capsules = currentGameState.getCapsules()
    distances = getDistances(capsules)
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    if sum(scaredTimes) or len(distances) == 0:
      return 0
    else:
      return 100 / (sum(distances) + 1) - len(capsules) * 300

  curScore = currentGameState.getScore()
  # print(curScore, ghostFeature(), foodFeature(), capsuleFeature())
  return curScore + ghostFeature() + foodFeature() + capsuleFeature()

  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

