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
               first = 'ApproximateAgent', second = 'ApproximateAgent'):
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
    Pogledati postavke za pokretanje u __init__ metodi
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.featExtractor = AdvancedExtractor()

        # Postavka za pokretanje vec treniranih agenata na 200 igara.
        # Potrebno je zakomentarisati i random selekciju akcija
        self.weights = {'bias': 1.3092480821245795, 'closest-food': -26.129769321065581, 'carrying-food': -17.22975776300706, 'invader-distances': -20.015589895371805, 'can-eat-enemy': 50.44534789948978, 'enemy-eaten': 117.25849008034925, 'non-invader-distances': 10.564552894428572, 'returns-food-home': 183.781861600454022, 'can-get-eaten': 2.075132783147235, 'eats-food': 46.07694128330801, 'home-distance': -25.3568123111908720}
        
        # Postavka tezina za dugotrajno treniranje
        # self.weights['bias'] = random.randrange(-5, 6, 1)
        # self.weights['closest-food'] = random.randrange(-5, 6, 1)
        # self.weights['carrying-food'] = random.randrange(-5, 6, 1)
        # self.weights['invader-distances'] = random.randrange(-5, 6, 1)
        # self.weights['can-eat-enemy'] = random.randrange(-5, 6, 1)
        # self.weights['enemy-eaten'] = random.randrange(-5, 6, 1)
        # self.weights['non-invader-distances'] = random.randrange(-5, 6, 1)
        # self.weights['returns-food-home'] = random.randrange(-5, 6, 1)
        # self.weights['can-get-eaten'] = random.randrange(-5, 6, 1)
        # self.weights['eats-food'] = random.randrange(-5, 6, 1)

        # Postavka tezina za kratkotrajno treniranje
        # self.weights = util.Counter()
        self.epsilon=0.1
        self.gamma=0.8
        self.alpha=0.2
        self.num_games = 0

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
        self.num_games += 1


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
        # print('calculate_action: ', time.time() - start_time)
        # actions = gameState.getLegalActions(self.index)
        # action = random.choice(actions)
        succ_state = gameState.generateSuccessor(self.index, action)
        # print('succ_state: ', time.time() - start_time)
        reward = self.calculate_reward(gameState, action, succ_state, self.index)
        # print('reward: ', time.time() - start_time)

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
        
        # prob = 0.9 - 0.8 * self.num_games / 250

        # if util.flipCoin(prob):
        #    return random.choice(legalActions)

        # if util.flipCoin(self.epsilon):
        #    return random.choice(legalActions)
        
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
        featureDict = self.featExtractor.getFeatures(state, action, self.index)
        for feature in featureDict:
            suma += self.weights[feature] * featureDict[feature]
        print('Q value: ', suma)
        return suma
    

    def calculate_reward(self, gameState, action, succGameState, agent):
        '''
        calculate_reward 2.0:
            Uzimamo u obzir koji je agent u pitanju, jer je ucenje agenata nezavisno
        '''
        reward = 0
        # Ovaj reward je ukoliko nije kraj igre
        if self.getPreviousObservation() is not None:
            reward += 20 * (self.getScore(succGameState) - self.getScore(gameState))
            if (reward != 0):
                print('stop')

        reward += 20 * (succGameState.getAgentState(agent).numCarrying - gameState.getAgentState(agent).numCarrying)
        if succGameState.getAgentState(agent).numCarrying - gameState.getAgentState(agent).numCarrying != 0:
            print('stop')

        reward += 100 * (succGameState.getAgentState(agent).numReturned - gameState.getAgentState(agent).numReturned)

        # Ovaj reward je ukoliko je nas agent pobedio
        if succGameState.isOver():
            reward = 100
        # Ovaj reward je ukoliko protivnicki agent u sledecem
        # potezu moze da pobedi
        for opponent in succGameState.getBlueTeamIndices():
            for action in succGameState.getLegalActions(opponent):
                succ_succ_state = succGameState.generateSuccessor(opponent, action)
                if (succ_succ_state.isOver()):
                    reward = -10

        # for opponent in succGameState.getBlueTeamIndices():
        #     print("POZICIJE: ", succGameState.getAgentPosition(agent), succGameState.getAgentPosition(opponent))
        #     t = (abs(succGameState.getAgentPosition(agent)[0] - succGameState.getAgentPosition(opponent)[0]), abs(succGameState.getAgentPosition(agent)[1] - succGameState.getAgentPosition(opponent)[1]))
        #     print(t)
        #     if not succGameState.getAgentState(opponent).isPacman and succGameState.getAgentState(opponent).scaredTimer == 0 and (t == (0,0) or t == (0,1) or t == (1,0) or t == (1, 1)):
        #         if succGameState.getAgentState(agent).numCarrying > 5:
        #             reward = -10
        #         else:
        #             reward = 0

        #     if not succGameState.getAgentState(agent).isPacman and succGameState.getAgentState(agent).scaredTimer == 0 and (t == (0,0) or t == (0,1) or t == (1,0) or t == (1, 1)):
        #         reward = 10
        
        # Rewards for eating agents
        agent_pos = succGameState.getAgentPosition(agent)
        agent_state = succGameState.getAgentState(agent)
        for enemy_index in gameState.getBlueTeamIndices():
            if self.getPreviousObservation() is None:
                continue
            enemy_pos = self.getPreviousObservation().getAgentPosition(enemy_index)
            next_enemy_pos = succGameState.getAgentPosition(enemy_index)
            t = (abs(next_enemy_pos[0] - enemy_pos[0]), abs(next_enemy_pos[1] - enemy_pos[1]))
            if not(t == (0, 1) or t == (1, 0) or t == (0, 0)):
                enemy_state = succGameState.getAgentState(enemy_index)
                # our ghost eaten their pacman
                if not agent_state.isPacman and agent_state.scaredTimer == 0:
                    reward = 40
                # our pacman eats their scared ghost
                if agent_state.isPacman and enemy_state.scaredTimer != 0:
                    reward = 45
            
                # # their pacman eats our scared ghost
                # if opp_state.isPacman and agent_state.scaredTimer != 0:
                #     reward = -10
                # # their ghost eats our pacman
                # if not opp_state.isPacman and opp_state.scaredTimer == 0:
                #     reward = -3 * agent_state.numCarrying
        # Reward for potential of getting eaten
        agent_state = succGameState.getAgentState(agent)
        agent_pos = succGameState.getAgentPosition(agent)
        walls = succGameState.getWalls()
        for enemy_index in succGameState.getBlueTeamIndices():
            enemy_pos = succGameState.getAgentPosition(enemy_index)
            enemy_state = succGameState.getAgentState(enemy_index)
            # t = (abs(agent_pos[0] - enemy_pos[0]), abs(agent_pos[1] - enemy_pos[1]))
            if agent_state.isPacman and enemy_state.scaredTimer == 0:
                # if t == (0, 1) or t == (1, 0) or t == (0, 0) or t == (1, 1):
                #     reward = min(-2, -2* agent_state.numCarrying)
                # elif t[0] + t[1] <= 5:
                #     reward = min(-2, -1 * agent_state.numCarrying)
                dist = distance(enemy_pos, agent_pos, walls)
                if dist <= 6 and dist > 3:
                    reward += min(-2, -2 * agent_state.numCarrying)
                elif dist <= 3:
                    reward += min(-3, -3 * agent_state.numCarrying)
            elif not agent_state.isPacman and enemy_state.isPacman and agent_state.scaredTimer == 0:
                dist = distance(enemy_pos, agent_pos, walls)
                if dist < 8 and dist > 4:
                    reward += 15
                elif dist <= 4:
                    reward += 20
            elif not agent_state.isPacman and not enemy_state.isPacman:
                # SAFE OPCIJA ZA BEG
                dist = distance(enemy_pos, agent_pos, walls)
                if dist <= 3:
                    old_agent_pos = gameState.getAgentPosition(agent)
                    if (old_agent_pos[0] == walls.width / 2) and (agent_pos[0] == walls.width / 2 - 1):
                        reward += 20

                   # OLD
        # for opponent in succGameState.getBlueTeamIndices():
        #     opp_pos = succGameState.getAgentPosition(opponent)
        #     t = (abs(agent_pos[0] - opp_pos[0]), abs(agent_pos[1] - opp_pos[1]))
        #     if t == (0, 1) or t == (1, 0) or t == (0, 0) or t == (1, 1):
        #         opp_state = succGameState.getAgentState(opponent)
        #         # our ghost eats their pacman
        #         if not agent_state.isPacman and agent_state.scaredTimer == 0:
        #             reward = 25
        #         # our pacman eats their scared ghost
        #         if agent_state.isPacman and opp_state.scaredTimer != 0:
        #             reward = 30
        #         # their pacman eats our scared ghost
        #         if opp_state.isPacman and agent_state.scaredTimer != 0:
        #             reward = -10
        #         # their ghost eats our pacman
        #         if not opp_state.isPacman and opp_state.scaredTimer == 0:
        #             reward = -3 * agent_state.numCarrying
            


        # for capsule in succGameState.getBlueCapsules():
        #     if succGameState.getAgentPosition(agent) == capsule:
        #         reward += 50

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
        if reward != 0:
            print(' ')
        difference = observedVal - self.getQValue(state, action)
        featureDict = self.featExtractor.getFeatures(state, action, self.index)
        print('Before: ' + str(self.weights))
        for feature in featureDict:
            self.weights[feature] = self.weights[feature] + self.alpha * difference * featureDict[feature]
        print("UPDATE")
        print('Reward: ' + str(reward) + ' Difference: ' + str(difference))
        print('After: ' + str(self.weights))

class SimpleExtractor:
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action, agent):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getBlueFood()
        walls = state.getWalls()
        # ghosts = state.getGhostPositions()
        blue_team = state.getBlueTeamIndices()
        enemies = [state.getAgentPosition(index) for index in blue_team]

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        # x, y = state.getPacmanPosition()
        x, y = state.getAgentPosition(agent)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Compute distance to closest ghost
        # Distanca od 1 znaci najveca udaljenost
        # Agent treba da podesi tezine da vrednuju ovaj feature:
        # Da ukoliko je ghostDistance velik, njegov uticaj na Q bude 
        # pozitivan, tj. ukoliko je mali (duh je blizu) da uticaj bude negativan
        # Tako agent bira stanja kod kojih je ghostDistance sto veci
        features['GhostDistance' + str(agent)] = 1 
        opponentsState = []
        for i in state.getBlueTeamIndices():
            opponentsState.append(state.getAgentState(i))
        visible = []
        for op in opponentsState:
            if not op.isPacman:
                visible.append(op)
        if len(visible) > 0:
            positions = [agent.getPosition() for agent in visible]
            closest = min(positions, key=lambda xx: manhattanDistance((x, y), xx))
            closestDist = manhattanDistance((x, y), closest)
            print("DUH")
            print(closestDist)
            if closestDist <= 10:
                print("POVECAJ ZA weight * ", closestDist/10) # delimo sa 100 da feature bude manji od 1
                features['GhostDistance' + str(agent)] = closestDist/10 

        # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        features["#-of-enemies-1-step-away"] += sum((next_x, next_y) in Actions.getLegalNeighbors(e, walls) for e in enemies)

        # if there is no danger of ghosts then add the food feature
        if food[next_x][next_y]:
            features["eats-food" + str(agent)] += 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food" + str(agent)] = 1 - 10 * float(dist) / (walls.width * walls.height)

        # foodList = food.asList()
        # features['successorScore'] = -len(foodList)

        nextState2 = state.generateSuccessor(agent, action)
        features["carryingFood" + str(agent)] = state.getAgentState(agent).numCarrying
        # if features["carryingFood"] > 5:
        #     features["closest-food"] = 0

        capsulesChasing = state.getBlueCapsules()
        capsulesChasingDistances = [manhattanDistance((x, y), capsule) for capsule in capsulesChasing]
        minCapsuleDistance = min(capsulesChasingDistances) if len(capsulesChasingDistances) else 0
        features["distanceToCapsule" + str(agent)] = minCapsuleDistance / 100 # da bude 0 < feature < 1

        enemiesAgents = [state.getAgentState(i) for i in state.getBlueTeamIndices()]
        invaders = [a for a in enemiesAgents if a.isPacman]
        # features['numInvaders'] = 1 / len(invaders) if len(invaders) != 0 else 1 # sto je manji broj to je gore - inverzan feature
        if len(invaders) > 0:
            dists = [manhattanDistance((x, y), a.getPosition()) for a in invaders]
            features['invaderDistance' + str(agent)] = 1 - min(dists) / 100 # da bude 0 < feature < 1
            features['invaderDistance' + str(agent)]/=10

        ###########################################
        # Feature: 'scores'
        # Agent indeksa 'agent' ce u narednom potezu postici rezultat
        # Opseg vrednosti: 0.00 ili 1.00
        ###########################################
        nextState = state.generateSuccessor(agent, action)
        features['scores' + str(agent)] = 1.0 if nextState.getScore() > state.getScore() else 0.0


        #features.divideAll(10.0)
        return features

class AdvancedExtractor:
    """
    Vraća osobine stanja koje bi trebalo da budu relevatne za Pacmana
    Osobine:
        1. 
    """

    def getFeatures(self, state, action, agent):
        features = util.Counter()

        ###### Feature 1: bias 
        features["bias"] = 1.0
        ###################################################

        ###### Feature 2: closest-food
        food = state.getBlueFood()
        walls = state.getWalls()
        x, y = state.getAgentPosition(agent)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = 10 * float(dist) / (walls.width * walls.height)

        if food[next_x][next_y]:
            features['eats-food'] = 1.0

        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(e, walls) for e in ghosts)
        ###################################################

        ###### Feature 3: food-carrying
        next_state = state.generateSuccessor(agent, action)
        features["carrying-food"] = next_state.getAgentState(agent).numCarrying / 20
        ###################################################

        ###### Feature 4: food-carrying-enemies
        # for enemy_index in state.getBlueTeamIndices():
        #     features["carrying-food-enemies"] += next_state.getAgentState(enemy_index).numCarrying / 20
        ###################################################

        ###### Feature 5: invader-distances
        ###### Feature 6: non-invader-distances
        ###### Feature 7: #-of-invaders
        agent_state = next_state.getAgentState(agent)
        for enemy_index in state.getBlueTeamIndices():
            enemy_state = next_state.getAgentState(enemy_index)
            enemy_pos = next_state.getAgentPosition(enemy_index)
            enemy_md = distance((next_x, next_y), enemy_pos, walls)
            if enemy_state.isPacman and not agent_state.isPacman:
                features['invader-distances'] += 1.0 / len(state.getBlueTeamIndices()) * \
                (10 * float(enemy_md) / (walls.width * walls.height))
            elif not enemy_state.isPacman and agent_state.isPacman:
                features['non-invader-distances'] += 1.0 / len(state.getBlueTeamIndices()) * \
                (10 * float(enemy_md) / (walls.width * walls.height))
        ###################################################

        ###### Feature 8: #-of-us-our-terit
        # for friend_index in state.getRedTeamIndices():
        #     # TODO prepisati da koristi agentState
        #     friend_pos = next_state.getAgentPosition(friend_index)
        #     flag = True if friend_pos[0] <= (walls.width / 2) else False
        #     features["#-of-us-our-terit"] += 1.0 / len(state.getBlueTeamIndices()) if flag else 0.0
        ###################################################

        ###### Feature 9: returns-food-home
        # TODO prepisati da koristi agentState
        if (x == walls.width / 2) and (next_x == walls.width / 2 - 1):
            features['returns-food-home'] = next_state.getAgentState(agent).numCarrying / 20
        ###################################################

        ###### Feature 10: eats-enemy
        # agent_state = next_state.getAgentState(agent)
        # for enemy_index in state.getBlueTeamIndices():
        #     enemy_pos = next_state.getAgentPosition(enemy_index)
        #     t = (abs(next_x - enemy_pos[0]), abs(next_y - enemy_pos[1]))
        #     if t == (0, 1) or t == (1, 0) or t == (0, 0) or t == (1, 1):
        #         if not agent_state.isPacman and agent_state.scaredTimer == 0:
        #             features['can-eat-enemy'] = 1.0
        #         if not agent_state.isPacman and agent_state.scaredTimer != 0:
        #             features['can-get-eaten'] = 1.0
        ###################################################

        ##### Feature 11: enemy-eaten
        for enemy_index in state.getBlueTeamIndices():
            enemy_pos = state.getAgentPosition(enemy_index)
            next_enemy_pos = next_state.getAgentPosition(enemy_index)
            t = (abs(next_enemy_pos[0] - enemy_pos[0]), abs(next_enemy_pos[1] - enemy_pos[1]))
            if not(t == (0, 1) or t == (1, 0) or t == (0, 0)):
                features['enemy-eaten'] = 1.0
                # TODO razlika da li je nas duh pojeo pakmena 
                # il je nas pakmen pojeo uplasenog duha
        ###################################################

        ###### Feature 12: can-get-eaten
        # prev_state = self.getPreviousObservation()
        # if prev_state is not None:
        #     prev_pos = prev_state.getAgentPosition(agent)
        #     pos = next_state.getAgentPosition(agent)
        #     t = (abs(pos[0] - prev_pos[0]), abs(pos[1] - prev_pos[1]))
        #     if not(t == (0, 1) or t == (1, 0) or t == (0, 0)):
        #         features['got-eaten'] = 1.0

        agent_state = next_state.getAgentState(agent)
        agent_pos = next_state.getAgentPosition(agent)
        for enemy_index in state.getBlueTeamIndices():
            enemy_pos = next_state.getAgentPosition(enemy_index)
            enemy_state = next_state.getAgentState(enemy_index)
            if agent_state.isPacman and not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                dist = distance(enemy_pos, agent_pos, walls)
                if dist >= 7:
                    features['non-invader-distances'] = 0.0
                else:
                    if dist <= 6 and dist > 3:
                        # features['can-get-eaten'] = 0.8
                        features['closest-food'] = 0.0
                        dist = distance((walls.width/2 - 1, agent_pos[1]), agent_pos, walls)
                        features['home-distance'] = 10 * float(dist) / (walls.width * walls.height)
                    elif dist <= 3:
                        features['closest-food'] = 0.0
                        dist = distance((walls.width/2 - 1, agent_pos[1]), agent_pos, walls)
                        features['home-distance'] = 10 * float(dist) / (walls.width * walls.height)
                    
                # if dist < 8 and dist > 4:
                #     # features['can-get-eaten'] = 0.8
                #     features['closest-food'] = 0.0
                # elif dist <= 4:
                #     # features['can-get-eaten'] = 1.0
                #     features['closest-food'] = 0.0
            elif not agent_state.isPacman and enemy_state.isPacman and agent_state.scaredTimer == 0:
                dist = distance(enemy_pos, agent_pos, walls)
                if dist < 30:
                    # features['can-eat-enemy'] = 0.2
                    features['closest-food'] = 0.0
                if dist < 20:
                    # features['can-eat-enemy'] = 0.5
                    features['closest-food'] = 0.0
                elif dist < 8 and dist > 4:
                    # features['can-eat-enemy'] = 0.8
                    features['closest-food'] = 0.0

                elif dist <= 4:
                    # features['can-eat-enemy'] = 1.0
                    features['closest-food'] = 0.0
            # RETURNING HOME WITHOUT REWARD
            elif not agent_state.isPacman and not enemy_state.isPacman:
                # SAFE OPCIJA ZA BEG
                dist = distance(enemy_pos, agent_pos, walls)
                if dist <= 3:
                    if (x == walls.width / 2) and (next_x == walls.width / 2 - 1):
                        features['returns-food-home'] = 1.0
                    else:
                        features['closest-food'] = 0.0


        ###################################################
        # TODO feature udaljenost od svoje polovine
        # TODO feature agent-eaten
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
            #print('closestFood', time.time() - start_time)
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    #print('closestFood', time.time() - start_time)
    return None

def distance(pos, pos2, walls):
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
        if (pos_x, pos_y) == pos2:
            #print('closestFood', time.time() - start_time)
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    #print('closestFood', time.time() - start_time)
    return None