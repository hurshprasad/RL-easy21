from src.episode import run
from numpy.random import random, randint
import numpy as np
from conf.constant import CONST
from utils.utility import init_dict
from utils.utility import write_dictionary_tofile as write
from utils.utility import plot_value_function as plot
from utils.utility import save_data as save

"""
 Monte Carlo Control:
    - Under MDP with no prior knowledge of transitions
    - learns from complete episodes
    - model free π′(s) = argmax Q(s, a)
    - model free prediction means estimate value function of unknown MDP
    - ε-greedy exploration εt = N0 /(N0 + mina N (st , a))
    - N(s,a) is the number of times that action a has been selected from state s
    - value = mean return
    - all episodes must terminate
    - Output: Optimal value function V∗
    - Output: Optimal policy π∗
"""


class MonteCarlo(object):
    N0 = 10
    Q = init_dict()
    N = init_dict()
    full_backup = True

    def action(self, state):

        rand = random()
        min_action = self.get_greedy(state)
        N = self.N["%d:%d:%d" % (state.dealer, state.player, min_action)]
        epsilon = self.N0/(self.N0 + N)

        if rand < epsilon:
            return randint(CONST.HIT, CONST.STICK + 1)
        else:
            return min_action

    def get_greedy(self, state):
        hit = self.N["%d:%d:%d" % (state.dealer, state.player, CONST.HIT)]
        stick = self.N["%d:%d:%d" % (state.dealer, state.player, CONST.STICK)]
        return CONST.STICK if stick < hit else CONST.HIT

    def backup(self, visit, reward):
        self.N[visit] += 1
        alpha = 1.0 / self.N[visit]
        self.Q[visit] += alpha * (reward - self.Q[visit])

    def get_value_function(self, dealer, player):

        V = np.zeros((len(player), len(dealer)))

        for d in dealer:
            for p in player:
                visit_hit = "%d:%d:%d" % (d, p, CONST.HIT)
                visit_stick = "%d:%d:%d" % (d, p, CONST.STICK)

                if self.Q[visit_hit] > self.Q[visit_stick]:
                    V[p - 1, d - 1] = self.Q[visit_hit]
                else:
                    V[p - 1, d - 1] = self.Q[visit_stick]

        return V

    def __str__(self):
        return 'MC'

if __name__ == '__main__':
    mc = MonteCarlo()
    run(mc)
    write(CONST.OUTPUT_PATH + "checkQ", mc.Q)
    print("\nPlotting Value Function")
    plot(mc, r'Optimal Value $V^*$', CONST.OUTPUT_PATH + "MC_Optimal")

    print("\nSaving Monte Carlo Q Value")
    save(CONST.MCQ, mc.Q)
