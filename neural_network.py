import numpy as np

# receives two 10-size vectors, a, b, and returns the L2 loss value
def loss_function(a, b):
    return 1.0/2.0 * np.square(np.subtract(a, b))


def loss_gradient(a, b):
    return np.subtract(a, b)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


class NeuralNetwork:
    # sets up layers and creates random matrices for feed_forward
    def __init__(self, layerSize):
        self.layers = len(layerSize)
        self.layerSize = layerSize

        self.bias = [np.zeros(1) for i in range(self.layers - 1)]
        self.theta = [np.zeros(1) for i in range(self.layers - 1)]

        for i in range(self.layers - 1):
            self.bias[i] = np.random.uniform(-1., 1., (layerSize[i + 1]))
            self.theta[i] = np.random.uniform(-1., 1., (layerSize[i + 1], layerSize[i]))

    # returns the prediction for each digit
    def feed_forward(self, curr_layer):
        for i in range(self.layers - 1):
            curr_layer = sigmoid(np.add(np.matmul(self.theta[i], curr_layer), self.bias[i]))

        return curr_layer

    # trains a neural network using gradient descent
    def train(self, image_list, epochs, images_per_epoch, alpha):
        curr_epoch = 1

        # make the images normalized
        get_images = image_list.return_image_all() / 255.0
        get_labels = image_list.return_label_all()

        while curr_epoch <= epochs:
            v = np.random.randint(0, get_images.shape[0] - 1, images_per_epoch)
            b = np.array([get_images[i].flatten() for i in v])
            l = np.array([get_labels[i] for i in v])

            bias_edits = [np.zeros(self.bias[i].shape) for i in range(self.layers - 1)]
            theta_edits = [np.zeros(self.theta[i].shape) for i in range(self.layers - 1)]

            for i in range(self.layers - 1):
                bias_edits[i] = np.zeros(self.bias[i].shape)
                theta_edits[i] = np.zeros(self.theta[i].shape)

            accuracy = 0.0

            for i in range(images_per_epoch):
                response = [np.zeros(1) for j in range(self.layers)]
                response[0] = b[i]
                expected_labels = np.zeros(self.layerSize[-1])
                expected_labels[l[i]] = 1.0

                for j in range(self.layers - 1):
                    response[j + 1] = sigmoid(np.add(np.matmul(self.theta[j], response[j]), self.bias[j]))

                accuracy += np.argmax(response[-1]) == np.argmax(expected_labels)

                back_response = loss_gradient(response[-1], expected_labels)

                for j in range(self.layers - 2, -1, -1):
                    theta_edits[j] = np.add(theta_edits[j], np.outer(back_response, response[j]))
                    bias_edits[j] = np.add(bias_edits[j], back_response)
                    back_response = np.matmul(self.theta[j].T, back_response) * response[j] * (1. - response[j])

            for i in range(self.layers - 1):
                self.bias[i] -= bias_edits[i] * alpha / float(images_per_epoch)
                self.theta[i] -= theta_edits[i] * alpha / float(images_per_epoch)

            print("Finished with epoch #", curr_epoch, " with accuracy: ", float(accuracy*100.0) / images_per_epoch, '%')
            curr_epoch += 1

    # returns the model's prediction for the given image
    def predict_label(self, img):
        flatten_image = img.flatten() / 255.0

        label_predictions = self.feed_forward(flatten_image)

        return np.argmax(label_predictions)


