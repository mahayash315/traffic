import numpy
import matplotlib.pyplot as plt

def plot(Y, pred_Y):
	''' Plot the actual Y and predicted Y

    :param Y: n-by-m actual data
    :param pred_Y: n-by-m predicted data
	'''
	# count the data
	n = len(Y)		# the number of observations
	m = len(Y[0])	# the number of observation locations

	# create a Figure instance
	fig = plt.figure()

	# create axes
	ax = []
	for i in xrange(m):
		ax.append(fig.add_subplot(m,1,i+1))

	# plot the actual data
	Yt = Y.transpose()
	pred_Yt = pred_Y.transpose()
	for i in xrange(m):
		ax[i].plot(numpy.array(xrange(n)), Yt[i], "r.-")
		ax[i].plot(numpy.array(xrange(n)), pred_Yt[i], "b.-")

	plt.show()
