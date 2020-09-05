import numpy as np
from matplotlib import pyplot as plt

x = np.array([1,2,3,4,5])
y = 3*x
z = 4*x
X = [x,y,z]

for i in range(3):
	plt.figure()
	plt.plot(X[i])
	plt.savefig("thefuck"+str(i))