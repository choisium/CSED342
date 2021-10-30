# ID: 20160169 NAME: Choi Soomin
######################################################################################

from engine.const import Const
import util, math, random, collections


############################################################
# Problem 1: Warmup
def get_conditional_prob1(delta, epsilon, eta, c2, d2):
    """
    :param delta: [δ] is the parameter governing the distribution of the initial car's position
    :param epsilon: [ε] is the parameter governing the conditional distribution of the next car's position given the previos car's position
    :param eta: [η] is the parameter governing the conditional distribution of the sensor's measurement given the current car's position
    :param c2: the car's 2nd position
    :param d2: the sensor's 2nd measurement

    :returns: a number between 0~1 corresponding to P(C_2=c2 | D_2=d2)
    """
    # Problem 1a
    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)

    # P(c2|d2) \prop P(d2|c2)P(c2)
    # P(c2) = Sum_c1 Sum_c1 P(c2|c1)P(c1)

    def get_p_c1(c1):
        return delta if c1 == 0 else 1 - delta

    def get_p_c2_bar_c1(c1, c2):
        return 1 - epsilon if c1 == c2 else epsilon

    def get_p_c2(c2):
        sum = 0
        for c1 in [0, 1]:
            sum += get_p_c2_bar_c1(c1, c2) * get_p_c1(c1)
        return sum

    def get_p_d2_bar_c2(d2, c2):
        return 1 - eta if d2 == c2 else eta

    def get_c2_bar_d2(c2, d2):
        p_c2 = get_p_c2(c2)
        p_d_bar_c2 = get_p_d2_bar_c2(d2, c2)
        return p_c2 * p_d_bar_c2

    def normalization(p_c2_bar_d2, d2):
        sum = 0
        for c2 in [0, 1]:
            sum += get_c2_bar_d2(c2, d2)
        return p_c2_bar_d2 / sum

    return normalization(get_c2_bar_d2(c2, d2), d2)

    # END_YOUR_ANSWER


def get_conditional_prob2(delta, epsilon, eta, c2, d2, d3):
    """
    :param delta: [δ] is the parameter governing the distribution of the initial car's position
    :param epsilon: [ε] is the parameter governing the conditional distribution of the next car's position given the previos car's position
    :param eta: [η] is the parameter governing the conditional distribution of the sensor's measurement given the current car's position
    :param c2: the car's 2nd position
    :param d2: the sensor's 2nd measurement
    :param d3: the sensor's 3rd measurement

    :returns: a number between 0~1 corresponding to P(C_2=c2 | D_2=d2, D_3=d3)
    """
    # Problem 1b
    # BEGIN_YOUR_ANSWER (our solution is 17 lines of code, but don't worry if you deviate from this)

    # P(c2|d2,d3) = P(c2,d3|d2) / P(d3|d2) \prop P(d3|c2,d2) P(c2|d2) = P(d3|c2)P(c2|d2)
    # P(d3|c2) = Sum_c3 P(d3,c3|c2) = Sum_c3 P(d3|c3,c2)P(c3|c2) = Sum_c3 P(d3|c3)P(c3|c2)

    def get_p_c3_bar_c2(c2, c3):
        return 1 - epsilon if c2 == c3 else epsilon

    def get_p_d3_bar_c3(d3, c3):
        return 1 - eta if d3 == c3 else eta

    def get_p_d3_bar_c2(d3, c2):
        sum = 0
        for c3 in [0, 1]:
            sum += get_p_d3_bar_c3(d3, c3) * get_p_c3_bar_c2(c2, c3)
        return sum

    def get_p_c2_bar_d2_d3(c2, d2, d3):
        p_c2_bar_d2 = get_conditional_prob1(delta, epsilon, eta, c2, d2)
        return get_p_d3_bar_c2(d3, c2) * p_c2_bar_d2

    def normalization(p_c2_bar_d2_d3, d2, d3):
        sum = 0
        for c2 in [0, 1]:
            sum += get_p_c2_bar_d2_d3(c2, d2, d3)
        return p_c2_bar_d2_d3 / sum

    return normalization(get_p_c2_bar_d2_d3(c2, d2, d3), d2, d3)

    # END_YOUR_ANSWER


# Problem 1c
def get_epsilon():
    """
    return a value of epsilon (ε)
    """
    # Problem 1c
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)

    # to check!
    # delta = 0.1
    # epsilon = 0.5
    # eta = 0.6
    # c2 = 0
    # d2 = 0
    # d3 = 3
    # p_c2_bar_d2 = get_conditional_prob1(delta, epsilon, eta, c2, d2)
    # p_c2_bar_d2_d3 = get_conditional_prob2(delta, epsilon, eta, c2, d2, d3)
    # print(p_c2_bar_d2, p_c2_bar_d2_d3, p_c2_bar_d2 == p_c2_bar_d2_d3)

    return 0.5
    # END_YOUR_ANSWER


# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):

    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = (
            False  ### ONLY USED BY GRADER.PY in case problem 3 has not been completed
        )
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    ############################################################
    # Problem 2:
    # Function: Observe (update the probablities based on an observation)
    # -----------------
    # Takes |self.belief| and updates it based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard
    #                 deviation Const.SONAR_STD
    #
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Don't forget to normalize self.belief!
    ############################################################

    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)

        for i in range(self.belief.getNumRows()):
            for j in range(self.belief.getNumCols()):
                tileX = util.colToX(j)
                tileY = util.rowToY(i)
                dist = math.sqrt((agentX - tileX) ** 2 + (agentY - tileY) ** 2)
                emission = util.pdf(dist, Const.SONAR_STD, observedDist)
                posterior = self.belief.getProb(i, j)
                self.belief.setProb(i, j, posterior * emission)

        self.belief.normalize()

        # END_YOUR_ANSWER

    ############################################################
    # Problem 3:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb, which gives all
    #   ((oldTile, newTile), transProb) key-val pairs that you must consider.
    # - Other ((oldTile, newTile), transProb) pairs not in self.transProb have
    #   zero probabilities and do not need to be considered.
    # - util.Belief is a class (constructor) that represents the belief for a single
    #   inference state of a single car (see util.py).
    # - Be sure to update beliefs in self.belief ONLY based on the current self.belief distribution.
    #   Do NOT invoke any other updated belief values while modifying self.belief.
    # - Use addProb and getProb to manipulate beliefs to add/get probabilities from a belief (see util.py).
    # - Don't forget to normalize self.belief!
    ############################################################
    def elapseTime(self):
        if self.skipElapse:
            return  ### ONLY FOR THE GRADER TO USE IN Problem 2
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)

        new_belief = util.Belief(self.belief.numRows, self.belief.numCols, 0.0)
        for ((oldTile, newTile), transProb) in self.transProb.items():
            posterior = self.belief.getProb(*oldTile)
            new_belief.addProb(*newTile, posterior * transProb)

        new_belief.normalize()
        self.belief = new_belief

        # END_YOUR_ANSWER

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self):
        return self.belief


# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):

    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructer that initializes an ParticleFilter object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in a dict of defaultdict
        # self.transProbDict[oldTile][newTile] = probability of transitioning from oldTile to newTile
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a defaultdict from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    ############################################################
    # Problem 4 (part a):
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$.
    # This algorithm takes two steps:
    # 1. Reweight the particles based on the observation.
    #    Concept: We had an old distribution of particles, we want to update these
    #             these particle distributions with the given observed distance by
    #             the emission probability.
    #             Think of the particle distribution as the unnormalized posterior
    #             probability where many tiles would have 0 probability.
    #             Tiles with 0 probabilities (no particles), we do not need to update.
    #             This makes particle filtering runtime to be O(|particles|).
    #             In comparison, exact inference (problem 2 + 3), most tiles would
    #             would have non-zero probabilities (though can be very small).
    # 2. Resample the particles.
    #    Concept: Now we have the reweighted (unnormalized) distribution, we can now
    #             resample the particles and update where each particle should be at.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new particle.
    ############################################################
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)

        for (tile, count) in self.particles.items():
            tileX = util.colToX(tile[1])
            tileY = util.rowToY(tile[0])
            dist = math.sqrt((agentX - tileX) ** 2 + (agentY - tileY) ** 2)
            emission = util.pdf(dist, Const.SONAR_STD, observedDist)
            self.particles[tile] = count * emission

        new_particles = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            particleIndex = util.weightedRandomChoice(self.particles)
            new_particles[particleIndex] += 1

        self.particles = new_particles

        # END_YOUR_ANSWER
        self.updateBelief()

    ############################################################
    # Problem 4 (part b):
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Read |self.particles| (defaultdict) corresonding to time $t$ and writes
    # |self.particles| corresponding to time $t+1$.
    # This algorithm takes one step
    # 1. Proposal based on the particle distribution at current time $t$:
    #    Concept: We have particle distribution at current time $t$, we want to
    #             propose the particle distribution at time $t+1$. We would like
    #             to sample again to see where each particle would end up using
    #             the transition model.
    #
    # Notes:
    # - transition probabilities is now using |self.transProbDict|
    # - Use util.weightedRandomChoice() to sample a new particle.
    # - To pass the grader, you must loop over the particles using
    #       for tile in self.particles
    #   and call util.weightedRandomChoice() $once per particle$ on the tile.
    ############################################################
    def elapseTime(self):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)

        new_particles = collections.defaultdict(int)
        for (tile, count) in self.particles.items():
            if count == 0: continue
            for _ in range(count):
                particleIndex = util.weightedRandomChoice(self.transProbDict[tile])
                new_particles[particleIndex] += 1

        self.particles = new_particles

        # END_YOUR_ANSWER

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self):
        return self.belief
