import numpy as np
import matplotlib.pyplot as plt

def plot(Y, pred_Y=None, title='', block=True):
	''' Plot the actual Y and predicted Y

    :param Y: n-by-m actual data
    :param pred_Y: n-by-m predicted data
	'''
	# count the data
	n = len(Y)		# the number of observations
	m = len(Y[0])	# the number of observation locations
	#m=min(1,m)

	# create a Figure instance
	fig = plt.figure()

	# create axes
	ax = []
	for i in xrange(m):
		ax.append(fig.add_subplot(m,1,i+1))

	# plot the data
	Yt = Y.transpose()
	if pred_Y == None:
		for i in xrange(m):
			ax[i].plot(np.array(xrange(n)), Yt[i], "b.-")
	else:
		pred_Yt = pred_Y.transpose()
		for i in xrange(m):
			ax[i].plot(np.array(xrange(n)), Yt[i], "b.-")
			ax[i].plot(np.array(xrange(n)), pred_Yt[i], "r.-")

	plt.title(title)
	plt.show(block=block)

class Plotter:
	def __init__(self):
		self.fig = None
		self.ax = None
		self.plot_y = None
		self.plot_pred_y = None
		self.last_x = -1

	def drawFrame(self):
		if (self.fig == None):
			self.fig = plt.figure()
			plt.show(block=False)

	def appendFirst(self, y, pred_y=None):
		m = len(y)	# the number of observation locations

		# create axes
		ax = []
		for i in xrange(m):
			ax.append(self.fig.add_subplot(m,1,i+1))

		# plot the data
		plot_y = []
		plot_pred_y = []
		yt = y.transpose()
		if pred_y == None:
			for i in xrange(m):
				plot_y.extend(ax[i].plot(np.array([0]), yt[i], "b.-"))
		else:
			pred_yt = pred_y.transpose()
			for i in xrange(m):
				plot_y.extend(ax[i].plot(np.array([0]), yt[i], "b.-"))
				plot_pred_y.extend(ax[i].plot(np.array([0]), pred_yt[i], "r.-"))

		self.ax = ax
		self.plot_y = plot_y
		self.plot_pred_y = plot_pred_y
		self.last_x = 0

	def append(self, y, pred_y=None):
		if (self.fig == None):
			self.drawFrame()

		if (self.last_x == -1):
			self.appendFirst(y, pred_y)
		else:
			x = self.last_x + 1
			for i in xrange(len(self.ax)):
				plot_y = self.plot_y[i]
				# plot_pred_y = self.plot_pred_y[i] # FIXME: to be fixed later
				plot_y.set_xdata(np.append(plot_y.get_xdata(), x))
				plot_y.set_ydata(np.append(plot_y.get_ydata(), y[i]))
				# plot_pred_y.set_xdata(np.append(plot_pred_y.get_xdata(), x))
				# plot_pred_y.set_ydata(np.append(plot_pred_y.get_ydata(), y[i]))
			self.last_x = x

		plt.draw()