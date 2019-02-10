import sys
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ImagePack:
	def decode(self, name):
		f = open(name)

		magic = (struct.unpack('>i', f.read(4)))[0]
		images = (struct.unpack('>i', f.read(4)))[0]
		self.rows = (struct.unpack('>i', f.read(4)))[0]
		self.columns = (struct.unpack('>i', f.read(4)))[0]

		images = 1000

		self.image = np.zeros((images, self.rows, self.columns))

		for i in range(images):
			for j in range(self.rows):
				self.image[i][j] = [ord(k) for k in f.read(self.columns)]
	def decode_labels(self, name):
		f = open(name)

		magic = (struct.unpack('>i', f.read(4)))[0]
		images = (struct.unpack('>i', f.read(4)))[0]

		images = 1000

		self.label = [ord(k) for k in f.read(1)]

	def getTrainingSize(self):
		return len(self.image)
		
	def getImageSize(self):
		return (self.rows, self.columns)

	def display(self, imgi):
		im = self.image[imgi]

		matplotlib.interactive(True)

		plt.ion()
		plt.gray()
		plt.imshow(im)
		plt.show()

		if raw_input(''):
			sys.exit(0)

	def returnImage(self, imgi):
		im = self.image[imgi]
		
		return im


