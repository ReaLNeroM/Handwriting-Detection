import numpy as np

class NeuralNetwork:
	# receives two 10-size vectors, a, b, and returns 1/2*(a-b).^2
	def lossFunction(self, a, b):
		return 1.0/2.0 * np.square(np.subtract(a, b))

	def lossGradient(self, a, b):
		return np.subtract(a, b)

	def sigmoid(self, a):
		return 1.0 / (1.0 + np.exp(-a))

	# sets up layers and creates random matrices for feedforward
	def __init__(self, layerSize):
		self.layers = len(layerSize)
		self.layerSize = layerSize

		self.bias = [np.zeros(1) for i in range(self.layers - 1)]
		self.theta = [np.zeros(1) for i in range(self.layers - 1)]

		for i in range(self.layers - 1):
			self.bias[i] = np.random.uniform(-0.000001, 0.000001, (layerSize[i + 1]))
			self.theta[i] = np.random.uniform(-0.000001, 0.000001, (layerSize[i + 1], layerSize[i]))

	# returns a 10-size vector
	def feedForward(self, curr_layer):
		for i in range(self.layers - 1):
			curr_layer = self.sigmoid(np.add(np.matmul(self.theta[i], curr_layer), self.bias[i]))

		return curr_layer

	# trains a neural network using gradient descent
	def train(self, image_list, epochs, imagesPerEpoch, alpha):
		currEpoch = 1

		# make the images normalized
		getImages = image_list.returnImageAll() / 255.0
		getLabels = image_list.returnLabelAll()

		while currEpoch <= epochs:
			v = np.random.randint(0, getImages.shape[0] - 1, imagesPerEpoch)
			b = np.array([getImages[i].flatten() for i in v])
			l = np.array([getLabels[i] 			 for i in v])

			biasEdits = [np.zeros(self.bias[i].shape) for i in range(self.layers - 1)]
			thetaEdits = [np.zeros(self.theta[i].shape) for i in range(self.layers - 1)]

			for i in range(self.layers - 1):
				biasEdits[i] = np.zeros(self.bias[i].shape)
				thetaEdits[i] = np.zeros(self.theta[i].shape)

			fitness = np.zeros(10)

			for i in range(imagesPerEpoch):
				response = [np.zeros(1) for j in range(self.layers)]
				response[0] = b[i]
				expectedLabels = np.zeros(self.layerSize[-1])
				expectedLabels[l[i]] = 1.0

				for j in range(self.layers - 1):
					response[j + 1] = self.sigmoid(np.add(np.matmul(self.theta[j], response[j]), self.bias[j]))

				# print response[-1], expectedLabels
				fitness += self.lossFunction(response[-1], expectedLabels) / imagesPerEpoch

				backResponse = self.lossGradient(response[-1], expectedLabels)

				for j in range(self.layers - 2, -1, -1):
					thetaEdits[j] = np.add(thetaEdits[j], np.outer(backResponse, response[j]))
					biasEdits[j] = np.add(biasEdits[j], backResponse)
					backResponse = np.matmul(self.theta[j].T, backResponse) * response[j] * (1. - response[j])

			for i in range(self.layers - 1):
				self.bias[i] -= biasEdits[i] / (alpha * imagesPerEpoch)
				self.theta[i] -= thetaEdits[i] / (alpha * imagesPerEpoch)

			print "Finished with epoch #", currEpoch, " with fitness: ", ['%.5f' % v for v in fitness], " eq. ", np.dot(v, v)
			currEpoch += 1

	# returns the model's prediction for the given image
	def predictLabel(self, img):
		flatten_image = img.flatten() / 255.0

		predictedLabels = self.feedForward(flatten_image)

		return np.argmax(predictedLabels)


