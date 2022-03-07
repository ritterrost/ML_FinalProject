import numpy as np

q_dicts = [{}, {}, {}, {}, {}, {}]

q_dicts[0]["okay"] = 10

a = np.array([1,5])
b = np.arange(100).reshape(50,2)

for i in range(20):
  print(np.random.randint(6))

print([np.random.random(4) for _ in range(6)])