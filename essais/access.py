import numpy as np

v = np.arange(3*5)
print(f"{v=}")
print(f"{v.reshape(3,5)=}")
print(f"{v.reshape(5,3)=}")
ind = np.array((0,2,4))
print(f"{v.reshape(3,5)[:,ind]=}")
print(f"{v.reshape(5,3)[ind,:]=}")
