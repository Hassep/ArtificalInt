import numpy as np
import matplotlib.pyplot as plt

def solveN(N):
	for k in range(0,10):
		N = (8/.05**2)*np.log((4*((2*N)**10)+1)/.05)
		plt.scatter(k,N)
		print N
	plt.savefig('2.12.plot.pdf', bbox_inches='tight')
	plt.show()
	print N

solveN(1000)

