
class State(object):

    def __init__(self):
        player = 0
        dealer = 0

    def __str__(self):
        return "Player: %d Dealer %d" % (self.player, self.dealer)
