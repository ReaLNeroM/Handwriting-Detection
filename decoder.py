import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ImagePack:
    def __init__(self, image_data_location, label_data_location):
        self.decode_images(image_data_location)
        self.decode_labels(label_data_location)

    def decode_images(self, name):
        file_object = open(name, 'rb')

        magic = int.from_bytes(file_object.read(4), byteorder='big', signed=False) # pylint: disable=unused-variable
        images = int.from_bytes(file_object.read(4), byteorder='big', signed=False)
        self.rows = int.from_bytes(file_object.read(4), byteorder='big', signed=False)
        self.columns = int.from_bytes(file_object.read(4), byteorder='big', signed=False)

        self.image = np.zeros((images, self.rows, self.columns))

        print("Loading images...")

        for i in range(images):
            for j in range(self.rows):
                self.image[i][j] = list(file_object.read(self.columns))

        print("Loading finished!")

    def decode_labels(self, name):
        file_object = open(name, 'rb')

        magic = int.from_bytes(file_object.read(4), byteorder='big', signed=False) # pylint: disable=unused-variable
        images = int.from_bytes(file_object.read(4), byteorder='big', signed=False)

        self.label = list(file_object.read(images))

    def get_training_size(self):
        return len(self.image)

    def get_image_size(self):
        return self.rows, self.columns

    def display(self, img_index):
        image_object = self.image[img_index]

        matplotlib.interactive(True)

        plt.ion()
        plt.gray()
        plt.imshow(image_object)
        plt.show()

        if input(''):
            sys.exit(0)

    def return_image(self, img_index):
        return self.image[img_index]

    def return_label(self, label_index):
        return self.label[label_index]

    def return_image_all(self):
        return self.image

    def return_label_all(self):
        return self.label
