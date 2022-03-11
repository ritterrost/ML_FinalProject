import numpy as np
from sklearn.linear_model import LinearRegression

from .callbacks import Q_table_func
from train import A_TO_NUM, ALPHA, GAMMA


def Q_table(self):
    transition = self.transitions[-1]
    used_Q_value = self.Q_values[A_TO_NUM[transition.action]]
    new_Q_values = Q_table_func(self, transition.next_state)

    rhs = ALPHA * (transition.reward + GAMMA * np.max(new_Q_values) - used_Q_value)
    self.Q_dicts[A_TO_NUM[transition.action]][hash(str(transition.state))] = (
        used_Q_value + rhs
    )


class lin_reg_model:
    def __init__(self):
        self.isFit = False
        self.models = [LinearRegression] * 6
        print(self.models)

    def get_Q_values(self, features):
        print("hello q values")
        # if(self.isFit):
        #   return

    def lin_reg(self):
        pass


def reg_tree():
    pass


def dec_tree():
    pass
