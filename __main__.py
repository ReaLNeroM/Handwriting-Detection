import decoder
import neural_network
from random import randint

image_list = decoder.ImagePack()
image_list.decode('train-images.idx3-ubyte')
image_list.decode_labels('train-labels.idx1-ubyte')

imagesNum = image_list.getTrainingSize()
imageSize = image_list.getImageSize()
labels = 10

trainingEpochs = 300
imagesPerEpoch = 2000
alpha = 5.

input_neurons = imageSize[0] * imageSize[1]

# the second layer has sqrt(|input|*|output|) neurons. This is a good rule of thumb. 
neural_net = neural_network.NeuralNetwork((input_neurons, int((input_neurons * labels) ** 0.5), labels))
neural_net.train(image_list, trainingEpochs, imagesPerEpoch, alpha)

for i in range(20):
	currImage = randint(0, imagesNum - 1)
	print "I predicted this image is: ", neural_net.predictLabel(image_list.returnImage(currImage)), " expected ans: ", image_list.returnLabel(currImage)
	image_list.display(currImage)
	print

raw_input("Are you satisfied with the results?\n")
# I don't process your input. Guess why ...
