from src.environment import Environment
from src.state import State
from conf.constant import CONST
from utils.utility import write_dictionary_tofile as write
import numpy as np


def play():
    state = State()

    env = Environment()
    state.dealer = env.draw(initial=True)
    state.player = env.draw(initial=True)

    print(state)

    while state.player < CONST.STAY_VALUE \
            and not env.check_burst(state.player):

        env.step(state, CONST.HIT)
        print(state)

    if not env.check_burst(state.player):
        env.step(state, CONST.STICK)

    print(state, " Reward: ", env.check_reward(state))


def checkDraw():

    frequency = [0] * CONST.DECK_SIZE
    for i in range(0, CONST.SAMPLES - 1, 1):
        card = Environment.draw()
        if card > 0:
            frequency[card+9] += 1
        elif card < 1:
            frequency[abs(card)-1] += 1

    entries = dict()
    for i in range(0, len(frequency)):
        black_red = -1 if i < 10 else 1
        card = i + 1 if i < 10 else i - 9
        key = "%d:%d" % (card, black_red)
        entries[key] = "%.3f" % (float(frequency[i])/CONST.SAMPLES)

    write(CONST.OUTPUT_PATH + "checkDraw", entries)


def checkStepDealer():

    record_data = {key: np.zeros(1, CONST.STEPDTYPE)
                     for key in CONST.STEPRECORDPLAYS}

    s = State()
    e = Environment()

    for rd in record_data:
        index = rd.split('-')
        a = CONST.HIT if int(index[2]) is 0 else CONST.STICK

        frequency = dict()
        for i in range(0, CONST.SAMPLES - 1, 1):
            s.dealer = int(index[0])
            s.player = int(index[1])
            r = e.step(s, a)

            key = "%d:%d:%d" % (s.dealer, s.player, r)
            if e.is_terminal(s):
                key = "%d:%d:%d" % (0, 0, r)

            if key in frequency:
                frequency[key] += 1
            else:
                frequency[key] = 1

        for entry in frequency:
            frequency[entry] = "%.3f" % (float(frequency[entry])/CONST.SAMPLES)

        filename = CONST.CHECKSTEPTEMPLATE.format(str(index[0]), str(index[1]), str(index[2]))

        write(CONST.OUTPUT_PATH + filename, frequency)


if __name__ == '__main__':
    checkDraw()
    checkStepDealer()
