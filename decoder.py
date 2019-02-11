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

		self.image = np.zeros((images, self.rows, self.columns))

		print "Loading images..."

		for i in range(images):
			for j in range(self.rows):
				self.image[i][j] = [ord(k) for k in f.read(self.columns)]

		print "Loading finished!"

	def decode_labels(self, name):
		f = open(name)

		magic = (struct.unpack('>i', f.read(4)))[0]
		images = (struct.unpack('>i', f.read(4)))[0]

		self.label = [ord(k) for k in f.read(images)]

	def getTrainingSize(self):
		return len(self.image)
		
	def getImageSize(self):
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

	def returnImage(self, img):
		return self.image[img]

	def returnLabel(self, lab):
		return self.label[lab]

	def returnImageAll(self):
		return self.image

	def returnLabelAll(self):
		return self.label

