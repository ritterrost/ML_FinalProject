from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


class Plotting:
    def __init__(self):
        self.counter = 0
        self.survial_time = []
        self.t = []
        self.c = 0

    def store_data(self, t):
        self.counter += 1
        self.t.append(t)
        # self.survial_time = st

    def __del__(self):
        self.c += 1
        print("destructor called")
        if self.counter > 0:
            self.reward_over_time()
        else:
            print("no data to plot!")

    def reward_over_time(self):
        print(self.c)
        print(np.arange(self.counter))
        print(self.t)
        plt.plot(np.arange(self.counter), self.t)
        plt.show()
