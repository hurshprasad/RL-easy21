from src.tasks.mc import MonteCarlo
from utils.utility import init_dict
from utils.utility import load_data
from utils.utility import plot_learning_curve as plot
from src.episode import run
import numpy as np
from math import pow
from conf.constant import CONST


class TD(MonteCarlo):
    E = init_dict()
    mse = np.zeros(CONST.EPISODES)
    l = 0.0
    MCQ = load_data(CONST.MCQ)

    def __init__(self):
        self.full_backup = False

    def get_greedy(self, state):
        hit = self.N["%d:%d:%d" % (state.dealer, state.player, CONST.HIT)]
        stick = self.N["%d:%d:%d" % (state.dealer, state.player, CONST.STICK)]
        return CONST.STICK if stick > hit else CONST.HIT

    def backup(self, cur_visit, visits, reward):

        # get previous step
        prev_visit = visits[len(visits) - 1]

        self.N[prev_visit] += 1

        # update eligibility trace (accumulating)
        self.E[prev_visit] += 1.0

        # calculate delta
        delta = reward + self.Q[cur_visit] - self.Q[prev_visit]

        alpha = 1.0 / self.N[prev_visit]
        self.Q[prev_visit] += alpha * delta * self.E[prev_visit]
        self.E[prev_visit] *= CONST.LAMBDAS[self.l]

    def record_mse(self, episode):
        _mse = 0.0
        for dealer in range(1, 10):
            for player in range(1, CONST.EASY21):
                for action in [CONST.HIT, CONST.STICK]:
                    visit = "%d:%d:%d" % (dealer, player, action)
                    if _mse < 0.00000001 and not _mse == 0.0:
                        _mse = 0.0
                    if not _mse > 10000.0:
                        _mse += (self.Q[visit] - self.MCQ[visit])**2
                        #print("episode ", episode, "mse ", _mse)

        self.mse[episode] = _mse / 10.0 / 21.0 / 2.0

    def __str__(self):
        return 'TD'


if __name__ == '__main__':

    smoothness_iter = 10
    mse = np.zeros((len(CONST.LAMBDAS), CONST.EPISODES))
    smse = np.zeros((smoothness_iter, CONST.EPISODES))

    for l in range(len(CONST.LAMBDAS)):
        for k in range(smoothness_iter):
            td = TD()
            td.l = l
            print("\nRunning TD Sarsa Lambda = ",
                  CONST.LAMBDAS[l], " Smooth = ", k + 1)
            run(td)
            smse[k, :] = td.mse

        cur_mse = np.mean(smse, axis=0)
        mse[l, :] = cur_mse

    print("\n Saving MSE Plots ")
    plot(mse)
