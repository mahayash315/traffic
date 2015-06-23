import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
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
		self.xs = ( [], [] )	# xdata, pred_xdata
		self.ys = [ ([], []) for _ in xrange(m) ] # ydata, pred_ydata
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
		xdata, pred_xdata = self.xs
		xdata.append(x)
		if (pred_y != None):
			pred_xdata.append(x)
		self.last_x = x

		for i in xrange(len(self.axs)):
			ydata, pred_ydata = self.ys[i]
			ydata.append(y[i])
			if (pred_y != None):
				pred_ydata.append(pred_y[i])

		self.update()

	def update(self):
		for i in xrange(len(self.axs)):
			self.plots[i][0].set_data(self.xs[0], self.ys[i][0])
			self.plots[i][1].set_data(self.xs[1], self.ys[i][1])
			self.axs[i].relim()
			self.axs[i].autoscale_view()

		plt.draw()