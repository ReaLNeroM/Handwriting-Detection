import os
import decoder
import neural_network
from random import randint
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

image_list = decoder.ImagePack()
image_list.decode(dir_path + '/train-images.idx3-ubyte')
image_list.decode_labels(dir_path + '/train-labels.idx1-ubyte')

testset_size = image_list.get_training_size()
image_dimensions = image_list.get_image_size()
labels = 10

try:
	ff = open(dir_path + '/nn', 'rb')

	print ("Loading previous neural network...")
	neural_net = pickle.load(ff)
except IOError:
	print ("No neural network found. Training new network...")

	epochs = 300
	images_per_epoch = 2000
	alpha = 5.

	input_neurons = image_dimensions[0] * image_dimensions[1]

	# the second layer has sqrt(|input|*|output|) neurons. This is a good rule of thumb. 
	neural_net = neural_network.NeuralNetwork((input_neurons, int((input_neurons * labels) ** 0.5), labels))
	neural_net.train(image_list, epochs, images_per_epoch, alpha)

ff = open(dir_path + '/nn', 'wb+')
pickle.dump(neural_net, ff)

correct = 0

for i in range(testset_size):
	correct += neural_net.predict_label(image_list.return_image(i)) == image_list.return_label(i)

print ('Final Accuracy:', str(float(correct * 100.0) / testset_size) + '%')

for i in range(20):
	curr_image = randint(0, testset_size - 1)
	prediction = neural_net.predict_label(image_list.return_image(curr_image))
	expected = image_list.return_label(curr_image)
	print ("I predicted this image is:", prediction, \
				", Expected ans:", expected, "correct? " + str(prediction == expected))
	image_list.display(curr_image)
	print ()

input("Are you satisfied with the results?\n")
# I don't process your input. Guess why ...
