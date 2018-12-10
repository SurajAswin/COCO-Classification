

import tensorflow as tf
import numpy as np


def get_bilinear_filter(filter_shape, upscale_factor):
	"""
	Creates a weight matrix that performs a bilinear interpolation
	:param filter_shape:   shape of the upscaling filter
	:param upscale_factor: scaling factor
	:return:               weight tensor
	"""
	kernel_size = filter_shape[1]

	if kernel_size % 2 == 1:
		centre_location = upscale_factor - 1
	else:
		centre_location = upscale_factor - 0.5

	bilinear = np.zeros([filter_shape[0], filter_shape[1]])
	for x in range(filter_shape[0]):
		for y in range(filter_shape[1]):
			value = (1 - abs((x - centre_location)/upscale_factor)) * \
			(1 - abs((y - centre_location)/upscale_factor))
			bilinear[x, y] = value

	weights = np.zeros(filter_shape)
	for i in range(filter_shape[2]):
		weights[:, :, i, i] = bilinear
	
	return weights
