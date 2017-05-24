from src.tasks.td import TD
from conf.constant import CONST
from src.episode import run
from utils.utility import plot_learning_curve as plot
from numpy.random import random, randint
from src.state import State
import numpy as np

"""
    Linear Function Approximation
    Estimate Value Function with Qθ(s,a)
    Generalise from seen states to unseen
    Update θ using MC/TD

"""


class LinearApproximation(TD):
    # randomly initialise theta vector
    theta = np.random.rand(len(CONST.DEALER_FEATURES) * len(CONST.PLAYER_FEATURES) * 2)
    E = np.zeros(len(CONST.DEALER_FEATURES) * len(CONST.PLAYER_FEATURES) * 2)

    def action(self, state):
        rand = random()

        if rand < CONST.EPSILON:
            return randint(CONST.HIT, CONST.STICK + 1)
        else:
            return self.get_greedy(state)

    def get_greedy(self, state):

        # get features per action
        F_hit = self.get_coarse_feature(state, CONST.HIT)
        F_stick = self.get_coarse_feature(state, CONST.STICK)

        # get value function
        Q_hit = np.sum(F_hit * self.theta)
        Q_stick = np.sum(F_stick * self.theta)

        return CONST.STICK if Q_stick > Q_hit else CONST.HIT

    def get_coarse_feature(self, state, action):

        features = np.zeros(len(CONST.DEALER_FEATURES) * len(CONST.PLAYER_FEATURES) * 2)

        if not state.player == state.dealer == 0:
            d = [1 if state.dealer in j else 0 for j in CONST.DEALER_FEATURES].index(1)

            # capture overlapping features
            for feature in CONST.PLAYER_FEATURES:
                if state.player in feature:
                    features[d * 11 + CONST.PLAYER_FEATURES.index(feature) * 2 + action] = 1

        return features

    def backup(self, cur_visit, visits, reward):

        # get previous state
        prev_visit = visits[len(visits) - 1]
        state_action = list(map(int, prev_visit.split(':')))
        state = State()
        state.dealer = state_action[0]
        state.player = state_action[1]
        action = state_action[2]

        # update eligibility trace (accumulating)

        Fv = self.get_coarse_feature(state, action)
        self.E[Fv == 1] += 1.0
        delta = reward - np.sum(Fv * self.theta)

        state_action = list(map(int, cur_visit.split(':')))
        state.dealer = state_action[0]
        state.player = state_action[1]
        action = state_action[2]

        Fv = self.get_coarse_feature(state, action)
        delta += np.sum(Fv * self.theta)
        self.E *= CONST.LAMBDAS[self.l]

        # update theta
        self.theta += CONST.ALPHA * delta * self.E

    def record_mse(self, episode):
        _mse = 0.0
        for dealer in range(1, 10):
            for player in range(1, CONST.EASY21):
                for action in [CONST.HIT, CONST.STICK]:
                    visit = "%d:%d:%d" % (dealer, player, action)
                    if _mse < 0.00000001 and not _mse == 0.0:
                        _mse = 0.0
                    if not _mse > 10000.0:
                        state = State()
                        state.dealer = dealer
                        state.player = player
                        F = self.get_coarse_feature(state, action)
                        Q = np.sum(F * self.theta)
                        _mse += (Q - self.MCQ[visit])**2

        self.mse[episode] = _mse / 10.0 / 21.0 / 2.0

    def __str__(self):
        return 'LFA'

if __name__ == '__main__':

    smoothness_iter = 2
    mse = np.zeros((len(CONST.LAMBDAS), CONST.EPISODES))
    smse = np.zeros((smoothness_iter, CONST.EPISODES))

    for l in range(len(CONST.LAMBDAS)):
        la = LinearApproximation()
        la.l = l
        for k in range(smoothness_iter):
            print("\nRunning Linear Function Approximation Lambda = ", CONST.LAMBDAS[l],
                  " smooth = ", k + 1)
            run(la)
            smse[k, :] = la.mse

        cur_mse = np.mean(smse, axis=0)
        mse[l, :] = cur_mse

    print("\n Saving MSE Plots ")
    plot(mse)
