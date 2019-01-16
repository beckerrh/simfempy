import numpy as np
import matplotlib.pyplot as plt
import splipy

c = splipy.curve_factory.circle()
t = np.linspace(c.start(0), c.end(0), 50)
x = c(t)

print("c", c)
print("c", help(c))
print("x.shape", x.shape)

plt.plot(x[:,0], x[:,1])
plt.show()