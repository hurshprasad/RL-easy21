import numpy as np


def constant(f):

    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()
    return property(fget, fset)


class _Const(object):

    @constant
    def REWARD():
        return 1

    @constant
    def DRAW():
        return 0

    @constant
    def HIT():
        return 0

    @constant
    def STICK():
        return 1

    @constant
    def STAY_VALUE():
        return 17

    @constant
    def EASY21():
        return 21

    @constant
    def DECK_SIZE():
        return 20

    @constant
    def SAMPLES():
        return 1000

    @constant
    def EPISODES():
        return 10000

    @constant
    def EPSILON():
        return .1

    @constant
    def ALPHA():
        return .01

    @constant
    def BATCH():
        return 100

    @constant
    def LAMBDAS():
        return [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    @constant
    def STEPRECORDPLAYS():
        return ['1-1-0', '1-10-0', '1-18-1', '10-15-1']

    @constant
    def DEALER_FEATURES():
        return [{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}]

    @constant
    def PLAYER_FEATURES():
        return [{1, 2, 3, 4, 5, 6},
                {4, 5, 6, 7, 8, 9},
                {7, 8, 9, 10, 11, 12},
                {10, 11, 12, 13, 14, 15},
                {13, 14, 15, 16, 17, 18},
                {16, 17, 18, 19, 20, 21}]

    @constant
    def CHECKSTEPTEMPLATE():
        return "checkStepDealer{:}Player{:}Action{:}"

    @constant
    def STEPDTYPE():
        return np.dtype([('dealer', np.int), ('player', np.int), ('reward', np.int)])

    @constant
    def DATAPATH():
        return "/home/hursh/resubmission/data/"
        #return "/Users/hurshprasad/UCL-Couseworks/Advanced Topics in Machine Learning/re_submission/data/"

    @constant
    def OUTPUT_PATH():
        return CONST.DATAPATH + "output/"

    @constant
    def MCQ():
        return CONST.DATAPATH + "output/mcq.data"


CONST = _Const()
