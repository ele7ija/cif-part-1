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
import distanceCalculator
import random
from game import Directions
import game
from game import Actions


#################
# Team creation #
#################

# B: Ovaj numTraining=2 je quick fix jer je bacao gresku kad nema ovog
#   keyword parametra
def createTeam(firstIndex, secondIndex, isRed, numTraining=2,
               first = 'ApproximateAgent', second = 'DummyAgent'):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_q()
    
    #######################################################
    # KOD SA VEZBI
    #######################################################
    def _init_q(self):
        self.Q = dict() 

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if state not in self.Q:
            self.Q[state] = dict()
            self.Q[state][action] = 0.0
        if action not in self.Q[state]:
            self.Q[state][action] = 0.0
        return self.Q[state][action]
        
    def setQValue(self, state, action, value):
        if state not in self.Q:
            self.Q[state] = dict()
        self.Q[state][action] = value
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        for action in state.getLegalActions(self.index):
            if best_action is None:
                best_action = action
            if self.getQValue(state, action) > self.getQValue(state, best_action):
                best_action = action
        if best_action is None:
            return 0.0
        return self.getQValue(state, best_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        for action in state.getLegalActions(self.index):
            if best_action is None:
                best_action = action
            if self.getQValue(state, action) > self.getQValue(state, best_action):
                best_action = action
        if best_action is None:
            return None
        return best_action

    def calculateAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        # action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        update = reward + self.gamma * self.computeValueFromQValues(nextState)
        newValue = self.getQValue(state, action)+self.alpha*(update - self.getQValue(state, action))
        self.setQValue(state, action, newValue)
        return self.getQValue(state, action)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    #######################################################
    # KRAJ KODA SA VEZBI
    #######################################################

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        
        self._init_params()

    def _init_params(self):
        self.epsilon=0.2
        self.alpha=0.5
        self.gamma=0.99

    def calculate_reward(self, gameState, action, succGameState):
        reward = 0
        # Ovaj reward je ukoliko nije kraj igre
        if self.getPreviousObservation() is not None:
            reward = self.getScore(succGameState) - self.getScore(succGameState)

        
        # Ovaj reward je ukoliko je nas agent pobedio
        if succGameState.isOver():
            reward = 100
        # Ovaj reward je ukoliko protivnicki agent u sledecem
        # potezu moze da pobedi
        for agent in succGameState.getBlueTeamIndices():
            for action in succGameState.getLegalActions(agent):
                succ_succ_state = succGameState.generateSuccessor(agent, action)
                if (succ_succ_state.isOver()):
                    reward = -100
        
        return reward

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        B: Za sada bira random, ali ubuduce treba da bira na osnovu Q-Vrednosti
        """
        start_time = time.time()
        action = self.calculateAction(gameState)
        # actions = gameState.getLegalActions(self.index)
        # action = random.choice(actions)
        succ_state = gameState.generateSuccessor(self.index, action)
        # print('succ_state: ', time.time() - start_time)
        reward = self.calculate_reward(gameState, action, succ_state)
        # print('reward: ', time.time() - start_time)

        if reward != 0:
            print('Reward: ', reward)
        self.update(gameState, action, succ_state, reward)
        # print('Ukupno vreme: ', time.time() - start_time)
        
        return action


class ExpectimaxAgent(CaptureAgent):
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
        self.depth = 1
        self.evaluationFunction = newEvaluationFunction

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        # actions = gameState.getLegalActions(self.index)
        start = time.time()
        bestVal = -1000
        bestAction = None
        print("======= NIVO NULA ======")
        for action in gameState.getLegalActions(self.index):
            newGameState = gameState.generateSuccessor(self.index, action)
            val = self.expectimax(newGameState, self.index+1, 1)
            print('Lap: ', time.time() - start)
            if val > bestVal:
                bestAction = action
                bestVal = val
        print("======= KRAJ NIVO NULA ========")
        print('Vreme ukupno: ', time.time() - start)
        return bestAction

        # return random.choice(actions)

    def expectimax(self, gameState, agent = 0, depth = 1):
        print(agent, depth)
        if agent == gameState.getNumAgents():
            agent = 0
            depth = depth + 1

        if gameState.isOver() or depth == self.depth + 1:
            return self.evaluationFunction(gameState)
            
        # VAZNO
        # Za pocetak samo crveni pobedjuju
        if agent in gameState.getRedTeamIndices():
            maxvr = -1000
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                vr = self.expectimax(succ, agent+1, depth)
                if vr > maxvr:
                    maxvr = vr
            return maxvr
        else:
            suma = 0.0
            i = 0
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                suma += self.expectimax(succ, agent+1, depth)
                i += 1
            return suma / i if i != 0 else 0

def scoreEvaluationFunction(currentGameState):
    """
      Ova default evaluaciona funkcija samo vraca skor stanja.
      Skor je isti onaj koji je ispisan na Pacman GUI-ju.

      Ova funkcija se koristi za agente sa protivnikom.
    """
    return currentGameState.getScore()

def newEvaluationFunction(currentGameState):
    """
      Funkcija evaluacije
    """
    score = 0

    newFood = currentGameState.getBlueFood()
    newGhosts = currentGameState.getBlueTeamIndices()
    newGhostStates = [currentGameState.getAgentState(ghost) for ghost in newGhosts]
    newCapsules = currentGameState.getBlueCapsules()
    for red in currentGameState.getRedTeamIndices():
        newPos = currentGameState.getAgentState(red).getPosition()
        closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if newCapsules:
            closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
        else:
            closestCapsule = 0

        if closestCapsule:
            closest_capsule = -3 / closestCapsule
        else:
            closest_capsule = 100

        if closestGhost:
            ghost_distance = -2 / closestGhost
        else:
            ghost_distance = -500

        foodList = newFood.asList()
        if foodList:
            closestFood = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            closestFood = 0

        score += -2 * closestFood + ghost_distance*10 - 10 * len(foodList) + closest_capsule

    print(score)
    return score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function.

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


###########################################################
### APPROXIMATION REINFORCEMENT LEARNING
###########################################################
class ApproximateAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.featExtractor = SimpleExtractor()
        self.weights = util.Counter()
        self.epsilon=0.1
        self.gamma=0.8
        self.alpha=0.2


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
        """
        Picks among actions randomly.
        B: Za sada bira random, ali ubuduce treba da bira na osnovu Q-Vrednosti
        """
        start_time = time.time()
        action = self.calculateAction(gameState)
        print('calculate_action: ', time.time() - start_time)
        # actions = gameState.getLegalActions(self.index)
        # action = random.choice(actions)
        succ_state = gameState.generateSuccessor(self.index, action)
        # print('succ_state: ', time.time() - start_time)
        reward = self.calculate_reward(gameState, action, succ_state)
        print('reward: ', time.time() - start_time)

        if reward != 0:
            print('Reward: ', reward)
        self.update(gameState, action, succ_state, reward)
        print('Ukupno vreme: ', time.time() - start_time)
        
        return action

    def calculateAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        # action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        
        return self.computeActionFromQValues(state)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_action_qvalue = None
        for action in state.getLegalActions(self.index):
            if best_action is None:
                best_action = action
                best_action_qvalue = self.getQValue(state, action)
                continue
            action_qvalue = self.getQValue(state, action)
            if action_qvalue > best_action_qvalue:
                best_action = action
                best_action_qvalue = action_qvalue
        if best_action is None:
            return 0.0
        return best_action_qvalue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_action_qvalue = None
        for action in state.getLegalActions(self.index):
            if best_action is None:
                best_action = action
                best_action_qvalue = self.getQValue(state, action)
                continue
            action_qvalue = self.getQValue(state, action)
            if action_qvalue > best_action_qvalue:
                best_action = action
                best_action_qvalue = action_qvalue
        if best_action is None:
            return None
        return best_action

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        start_time = time.time()
        suma = 0
        featureDict = self.featExtractor.getFeatures(state, action)
        for feature in featureDict:
            suma += self.weights[feature] * featureDict[feature]
        # print('\tgetQvalue', time.time() - start_time)
        return suma
    

    def calculate_reward(self, gameState, action, succGameState):
        reward = 0
        # Ovaj reward je ukoliko nije kraj igre
        if self.getPreviousObservation() is not None:
            reward = self.getScore(succGameState) - self.getScore(succGameState)

        for agent in succGameState.getRedTeamIndices():
            reward += 10 * succGameState.getAgentState(agent).numCarrying 

        # Ovaj reward je ukoliko je nas agent pobedio
        if succGameState.isOver():
            reward = 100
        # Ovaj reward je ukoliko protivnicki agent u sledecem
        # potezu moze da pobedi
        for agent in succGameState.getBlueTeamIndices():
            for action in succGameState.getLegalActions(agent):
                succ_succ_state = succGameState.generateSuccessor(agent, action)
                if (succ_succ_state.isOver()):
                    reward = -100
        
        return reward

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        observedVal = reward + self.gamma * self.computeValueFromQValues(state)
        difference = observedVal - self.getQValue(state, action)
        featureDict = self.featExtractor.getFeatures(state, action)
        print('Before: ' + str(self.weights))
        for feature in featureDict:
            self.weights[feature] = self.weights[feature] + self.alpha * difference * featureDict[feature]
        # print('Reward: ' + str(reward) + ' Difference: ' + str(difference))
        print('After: ' + str(self.weights))

class SimpleExtractor:
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getBlueFood()
        walls = state.getWalls()
        # ghosts = state.getGhostPositions()
        blue_team = state.getBlueTeamIndices()
        enemies = [state.getAgentPosition(index) for index in blue_team]

        features = util.Counter()

        features["bias"] = 1.0

        for friend in state.getRedTeamIndices():
            # compute the location of pacman after he takes the action
            # x, y = state.getPacmanPosition()
            x, y = state.getAgentPosition(friend)
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            # count the number of ghosts 1-step away
            # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
            features["#-of-enemies-1-step-away"] += sum((next_x, next_y) in Actions.getLegalNeighbors(e, walls) for e in enemies)

            # if there is no danger of ghosts then add the food feature
            if features["#-of-enemies-1-step-away"] == 0 and food[next_x][next_y]:
                features["eats-food"] += 1.0

            dist = closestFood((next_x, next_y), food, walls)
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] += float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    start_time = time.time()
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            print('closestFood', time.time() - start_time)
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    print('closestFood', time.time() - start_time)
    return None