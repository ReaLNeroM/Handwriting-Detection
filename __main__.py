import os
import decoder
import neural_network
from random import randint

dir_path = os.path.dirname(os.path.realpath(__file__))

image_list = decoder.ImagePack()
image_list.decode(dir_path + '/train-images.idx3-ubyte')
image_list.decode_labels(dir_path + '/train-labels.idx1-ubyte')

testset_size = image_list.get_training_size()
image_dimensions = image_list.get_image_size()
labels = 10

epochs = 3000
images_per_epoch = 200
alpha = 5.

input_neurons = image_dimensions[0] * image_dimensions[1]

# the second layer has sqrt(|input|*|output|) neurons. This is a good rule of thumb. 
neural_net = neural_network.NeuralNetwork((input_neurons, int((input_neurons * labels) ** 0.5), labels))
neural_net.train(image_list, epochs, images_per_epoch, alpha)

for i in range(20):
	curr_image = randint(0, testset_size - 1)
	print "I predicted this image is: ", neural_net.predict_label(image_list.return_image(curr_image)), \
				" expected ans: ", image_list.return_label(curr_image)
	image_list.display(curr_image)
	print

raw_input("Are you satisfied with the results?\n")
# I don't process your input. Guess why ...
