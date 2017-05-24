import random
from conf.constant import CONST


class Environment(object):

    @staticmethod
    def init_state(state):
        # Initialise state of game
        state.dealer = Environment.draw(initial=True)
        state.player = Environment.draw(initial=True)

    @staticmethod
    def terminal(state):
        return Environment.check_burst(state.player) \
               or Environment.check_burst(state.dealer) \
               or state.dealer >= CONST.STAY_VALUE

    @staticmethod
    def check_burst(score):
        return False if 1 <= score <= CONST.EASY21 else True

    @staticmethod
    def draw(initial=False):
        probability = random.random()
        card = random.randint(1, 10)
        if initial or probability <= float(2) / 3:
            return card
        else:
            return -card

    @staticmethod
    def check_reward(state):
        if Environment.check_burst(state.player):
            return -CONST.REWARD
        elif Environment.check_burst(state.dealer):
            return CONST.REWARD
        elif state.dealer < CONST.STAY_VALUE:
            return CONST.DRAW
        elif state.player > state.dealer:
            return CONST.REWARD
        elif state.player == state.dealer:
            return CONST.DRAW
        else:
            return -CONST.REWARD

    @staticmethod
    def is_terminal(state):

        if not 0 < state.dealer < CONST.STAY_VALUE \
                or Environment.check_burst(state.player):
            return True

        return False

    @staticmethod
    def step(state, action=CONST.HIT):

        if action == CONST.HIT:
            state.player += Environment.draw()
        else:
            while not Environment.terminal(state):
                state.dealer += Environment.draw()

        reward = Environment.check_reward(state)

        if Environment.is_terminal(state):
            state.player = state.dealer = 0

        return reward

