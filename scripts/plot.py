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
		self.axs = None
		self.plots = None
		self.xs = None
		self.ys = None
		self.last_x = -1

	def drawFrame(self):
		if (self.fig == None):
			self.fig = plt.figure()
			plt.show(block=False)

	def initDraw(self, m):
		self.xs = [ [], [] ]	# y, pred_y
		self.ys = [ [[], []] for _ in xrange(m) ]
		self.axs = []
		self.plots = []

		# create axes and plots
		for i in xrange(m):
			ax = self.fig.add_subplot(m,1,i+1)
			plot_y = ax.plot(self.xs[0], self.ys[i][0], "b.-")[0]
			plot_pred_y = ax.plot(self.xs[1], self.ys[i][1], "r.-")[0]
			self.axs.append(ax)
			self.plots.append([plot_y, plot_pred_y])

		plt.draw()

	def append(self, y, pred_y=None):
		if (self.fig == None):
			self.drawFrame()
			self.initDraw(len(y))

		x = self.last_x + 1
		self.xs[0].append(x)
		if (pred_y != None):
			self.xs[1].append(x)
		self.last_x = x

		for i in xrange(len(self.axs)):
			self.ys[i][0].append(y[i])
			if (pred_y != None):
				self.ys[i][1].append(pred_y[i])

		self.update()

	def update(self):
		for i in xrange(len(self.axs)):
			self.plots[i][0].set_data(self.xs[0], self.ys[i][0])
			self.plots[i][1].set_data(self.xs[1], self.ys[i][1])
			self.axs[i].relim()
			self.axs[i].autoscale_view()

		plt.draw()