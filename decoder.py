import sys
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ImagePack:
	def decode(self, name):
		f = open(name, 'rb')

		magic = int.from_bytes(f.read(4), byteorder='big', signed=False)
		images = int.from_bytes(f.read(4), byteorder='big', signed=False)
		self.rows = int.from_bytes(f.read(4), byteorder='big', signed=False)
		self.columns = int.from_bytes(f.read(4), byteorder='big', signed=False)

		self.image = np.zeros((images, self.rows, self.columns))

		print ("Loading images...")

		for i in range(images):
			for j in range(self.rows):
				self.image[i][j] = [k for k in f.read(self.columns)]

		print ("Loading finished!")

	def decode_labels(self, name):
		f = open(name, 'rb')

		magic = int.from_bytes(f.read(4), byteorder='big', signed=False)
		images = int.from_bytes(f.read(4), byteorder='big', signed=False)

		self.label = [k for k in f.read(images)]

	def get_training_size(self):
		return len(self.image)
		
	def get_image_size(self):
		return self.rows, self.columns

	def display(self, img):
		im = self.image[img]

		matplotlib.interactive(True)

		plt.ion()
		plt.gray()
		plt.imshow(im)
		plt.show()

		if raw_input(''):
			sys.exit(0)

	def return_image(self, img):
		return self.image[img]

	def return_label(self, lab):
		return self.label[lab]

	def return_image_all(self):
		return self.image

	def returnLabel_all(self):
		return self.label

