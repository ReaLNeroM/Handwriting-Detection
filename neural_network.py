import numpy as np

class NeuralNetwork:
	# receives two 10-size vectors, a, b, and returns 1/2*(a-b).^2
	def lossFunction(self, a, b):
		return 1.0/2.0 * np.sum(np.square(np.subtract(a, b)))
	def lossGradient(self, a, b):
		return np.subtract(a, b)
	def sigmoid(self, a):
		return 1.0 / (1.0 + np.exp(-a))
	def sigmoidGradient(self, a):
		return sigmoid(a) * (1.0 - sigmoid(a))

	# sets up layers and creates random matrices for feedforward
	def __init__(self, layerSize):
		self.layers = len(layerSize)
		self.layerSize = layerSize

		self.bias = [np.zeros(1) for i in range(self.layers - 1)]
		self.theta = [np.zeros(1) for i in range(self.layers - 1)]
		for i in range(self.layers - 1):
			self.bias[i] = np.random.normal(0.0, 1.0, (layerSize[i + 1]))
			self.theta[i] = np.random.normal(0.0, 1.0, (layerSize[i + 1], layerSize[i]))
			print self.bias[i], self.theta[i]

	# returns a 10-size vector
	def feedForward(self, curr_layer):
		for i in range(self.layers - 1):
			curr_layer = self.sigmoid(np.add(np.matmul(self.theta[i], curr_layer), self.bias[i]))

		return curr_layer

	# trains a neural network using gradient descent
	def train(self, image_list, epochs, imagesPerEpoch):
		currEpoch = 1

		while currEpoch <= epochs:

			#extract imagesPerEpoch images
			#perform gradient descent, sum it over all iterations
			#divide by imagesPerEpoch
			#perform changes
			print "Finished with epoch #", currEpoch
			currEpoch += 1

	# returns the model's prediction for the given image
	def predictLabel(self, img):
		flatten_image = img.flatten()

		predictedLabels = self.feedForward(flatten_image)

		print predictedLabels
		return np.argmax(predictedLabels)

