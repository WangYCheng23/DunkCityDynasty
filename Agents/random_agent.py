import numpy as np


class RandomAgent():
    def __init__(self):
        pass

    def get_actions(self, states):
        return {key: np.random.randint(0, 8) for key in states}