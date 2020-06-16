# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

# B: Ovaj numTraining=2 je quick fix jer je bacao gresku kad nema ovog
#   keyword parametra
def createTeam(firstIndex, secondIndex, isRed, numTraining=2,
               first = 'TabularQLearningAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


class TabularQLearningAgent(CaptureAgent):
  """
  An agent that utilizes Tabular Q-Learning 
  B:  Za sada bih probao sa tabularnim agentom. U odgovornosti agenta je da menja
      Q(State, Action) funkciju i nju koristi da odabere akciju.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def calculate_reward(self, gameState, action):
    reward = 0
    # Ovaj reward je ukoliko nije kraj igre
    if self.getPreviousObservation() is not None:
      reward = self.getScore(gameState) - self.getScore(self.getPreviousObservation())

    succ_state = gameState.generateSuccessor(self.index, action)
    # Ovaj reward je ukoliko je nas agent pobedio
    if succ_state.isOver():
      reward = 100
    # Ovaj reward je ukoliko protivnicki agent u sledecem
    # potezu moze da pobedi
    for agent in succ_state.getBlueTeamIndices():
      for action in succ_state.getLegalActions(agent):
        succ_succ_state = succ_state.generateSuccessor(agent, action)
        if (succ_succ_state.isOver()):
          reward = -100
    
    return reward

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    B: Za sada bira random, ali ubuduce treba da bira na osnovu Q-Vrednosti
    """
    # 1.  Posto nam nagrada nije data (u zadatku iz vezbi jeste), izracunaj 
    #     svojevoljno nagradu koja se dobija prelazom u drugo stanje.
    # 2.  Na osnovu nagrade, apdejtuj Q-vrednosti koje agent cuva
    # 3.  Izaberi akciju koja za stanje S ima najvecu Q vrednost (za sada random)
    actions = gameState.getLegalActions(self.index)
    action = random.choice(actions)
    print('Reward: ', self.calculate_reward(gameState, action))
    
    return action