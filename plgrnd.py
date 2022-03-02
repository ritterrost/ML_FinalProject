import numpy as np

field = np.arange(17 * 17).reshape(17, 17)
sight = 2

y = 1
x = 15

xx, yy = np.clip(np.mgrid[x - sight : x + sight, y - sight : y + sight], 0, 16)

rel_field = field[yy, xx]

bombs = [((5, 1), 1), ((5, 15), 2)]
bomb_xys = [list(xy) for (xy, t) in bombs]

betas = [[], [], [], [], [], []]
betas = [np.ones(57) for _ in betas]

size = 100

a = np.arange(2)
b = np.arange(2 * 57).reshape(2, 57)
print(np.random.randint(5))
