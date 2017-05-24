from src.environment import Environment
from src.state import State
from conf.constant import CONST


def run(model):

    # initialise state & environment
    state = State()

    visits = list()

    for episode in range(0, CONST.EPISODES):

        # Initialise state of game
        Environment.init_state(state)

        # clear all visits
        if model.full_backup:
            del visits[:]

        # for TD based tasks
        if not model.full_backup:
            action = model.action(state)

        # Play one game/episode until not Terminal
        while not Environment.terminal(state):

            # Capture State Visit per episode
            if not model.full_backup:
                visits.append("%d:%d:%d" % (state.dealer, state.player, action))
                reward = Environment.step(state, action)

            # Get Action
            action = CONST.STICK if Environment.terminal(state) \
                else model.action(state)

            # Capture All State Visit per Episode
            if model.full_backup:
                visits.append("%d:%d:%d" % (state.dealer, state.player, action))
                reward = Environment.step(state, action)

            # Backup Previous Visit
            if not model.full_backup:
                cur_visit = "%d:%d:%d" % (state.dealer, state.player, action)
                model.backup(cur_visit, visits, reward)
                del visits[:]

        print('\r' + str(model) + ' Episode...' + str(episode + 1), end='')

        if model.full_backup:
            # Record Visit/Reward
            for visit in visits:
                model.backup(visit, reward)

        if not model.full_backup:
            model.record_mse(episode)
