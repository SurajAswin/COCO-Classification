import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from upscale import get_bilinear_filter
DEBUG = True
class architecture:
	
	def __init__(self, tf_session, size, image_size, num_channels, cls, bx):

		self._sess = tf_session
		self.batch_size = size
		self.image_size = image_size
		self.channels = num_channels
		self.fc1_output = cls
		self.fc2_output = bx

	def initialise(self, sess):

		with tf.device('/gpu:0'):
			self.tf_input_vector = tf.placeholder(tf.float32, shape=(None, 512, 512, 3))
	
			#Convolutional layer 1, weights and biases
			self.hy_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 32],stddev = 0.1), name = 'hy_conv1_w')
			self.hy_conv1_biases = tf.Variable(tf.zeros([32]), name = 'hy_conv1_b')

			#Convolutional layer 1a, weights and biases
			self.hy_conv1a_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 32],stddev = 0.1), name = 'hy_conv1a_w')
			self.hy_conv1a_biases = tf.Variable(tf.zeros([32]), name = 'hy_conv1a_b')

			#Convolutional layer 2, weights and biases
			self.hy_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev = 0.1), name = 'hy_conv2_w')
			self.hy_conv2_biases = tf.Variable(tf.random_normal(shape = [64]), name = 'hy_conv2_b')

			#Convolutional layer 2a, weights and biases
			self.hy_conv2a_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.1), name = 'hy_conv2a_w')
			self.hy_conv2a_biases = tf.Variable(tf.random_normal(shape = [64]), name = 'hy_conv2a_b')

			#Convolutional layer 3, weights and biases
			self.hy_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev = 0.1), name = 'hy_conv3_w')
			self.hy_conv3_biases = tf.Variable(tf.random_normal(shape = [128]),  name = 'hy_conv3_b')

			#Convolutional layer 3a, weights and biases
			self.hy_conv3a_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev = 0.1), name = 'hy_conv3a_w')
			self.hy_conv3a_biases = tf.Variable(tf.random_normal(shape = [128]),  name = 'hy_conv3a_b')
			




			#Convolutional layer 4, weights and biases
			self.hy_conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 1], stddev = 0.1), name = 'hy_conv4_w')
			self.hy_conv4_biases = tf.Variable(tf.random_normal(shape = [1]), name = 'hy_conv4_b')




			#Convolutional layer 4, weights and biases
			self.hy_conv4a_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 1], stddev = 0.1), name = 'hy_conv4a_w')
			self.hy_conv4a_biases = tf.Variable(tf.random_normal(shape = [1]), name = 'hy_conv4a_b')


			#dense1
			self.hy_dense1_weights = tf.Variable(tf.truncated_normal([1, 1], stddev=0.01))
			self.hy_dense1_biases = tf.Variable(tf.random_normal(shape=[1]))
			# Dense layer
			self.hy_dense2_weights = tf.Variable(tf.truncated_normal([1, 1], stddev=0.01))
			self.hy_dense2_biases = tf.Variable(tf.random_normal(shape=[1]))

			self.hy_dense3_weights = tf.Variable(tf.truncated_normal([1, 1], stddev=0.01))
			self.hy_dense3_biases = tf.Variable(tf.random_normal(shape=[1]))

			self.hy_dense4_weights = tf.Variable(tf.truncated_normal([1, 1], stddev=0.01))
			self.hy_dense4_biases = tf.Variable(tf.random_normal(shape=[1]))

			
			#transpose
			self.kernel_size_0 = 2*2 - 2%2
			self.new_shape_0 = [1, 512, 512, 3]
			self.output_shape_0 = tf.stack(self.new_shape_0)

			self.kernel_size_1 = 2*4 - 4%2

			#transpose
			self.kernel_size_1 = 2*8 - 8%2
			self.new_shape = [1, 512, 512, 3]
			self.output_shape = tf.stack(self.new_shape)
			
		
			init = tf.global_variables_initializer()
			sess.run(init)


			def model(data):
		
				X = tf.reshape(data, shape = [-1, 512, 512, self.channels])
				if DEBUG == True : print('Shape X : ' + str(X.get_shape()))
				print(data)
				###########################################################################################################################################################################

				#Convolutional layer 1
				conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hy_conv1_weights, strides = [1,2,2,1], padding = 'SAME'), self.hy_conv1_biases))
				if DEBUG == True : print('Shape Conv1 : ' + str(conv1.get_shape()))

				#Convolutional layer 1a
				conv1a = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(conv1, self.hy_conv1a_weights, strides = [1,1,1,1], padding = 'SAME'), self.hy_conv1a_biases))
				if DEBUG == True : print('Shape Conv1a : ' + str(conv1a.get_shape()))
	
				pool1 = tf.nn.max_pool(conv1a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
				if DEBUG == True : print('Shape Pool1 : ' + str(pool1.get_shape())) 
				#norm1 = tf.nn.lrn(pool1, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)
	
				###########################################################################################################################################################################
	
				#Convolutional layer 2
				conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(pool1, self.hy_conv2_weights, strides = [1,2,2,1], padding = 'SAME'), self.hy_conv2_biases))
				if DEBUG == True : print('Shape Conv2 : ' + str(conv2.get_shape()))
	
				#Convolutional layer 2a
				conv2a = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(conv2, self.hy_conv2a_weights, strides = [1,1,1,1], padding = 'SAME'), self.hy_conv2a_biases))
				if DEBUG == True : print('Shape Conv2a : ' + str(conv2a.get_shape()))
	
				pool2 = tf.nn.max_pool(conv2a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
				if DEBUG == True : print('Shape Pool2 : ' + str(pool2.get_shape())) 
				norm2 = tf.nn.lrn(pool2, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)

				############################################################################################################################################################################
		
				#Convolutional layer 3
				conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hy_conv3_weights, strides = [1,2,2,1], padding = 'SAME'), self.hy_conv3_biases))
				if DEBUG == True : print('Shape Conv3 : ' + str(conv3.get_shape()))

				#Convolutional layer 3a
				conv3a = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(conv3, self.hy_conv3a_weights, strides = [1,2,2,1], padding = 'SAME'), self.hy_conv3a_biases))
				if DEBUG == True : print('Shape Conv3a : ' + str(conv3a.get_shape()))
		
				pool3 = tf.nn.max_pool(conv3a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
				if DEBUG == True : print('Shape Pool3 : ' + str(pool3.get_shape())) 
				norm3 = tf.nn.lrn(pool3, bias = 1.0, alpha = 0.001/9.0, beta = 0.75)
				
				############################################################################################################################################################################
		
				#Convolutional layer 4
				#conv4 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(pool3, self.hy_conv4_weights, strides = [1,64,64,1], padding = 'SAME'), self.hy_conv4_biases))
				#if DEBUG == True : print('Shape Conv4 : ' + str(conv4.get_shape()))

				############################################################################################################################################################################

				#Convolutional layer 4a
				conv4a = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm3, self.hy_conv4_weights, strides = [1,2,2,1], padding = 'SAME'), self.hy_conv4_biases))
				if DEBUG == True : print('Shape Conv4a : ' + str(conv4a.get_shape()))

				fc1 = tf.reshape(conv4a, [self.hy_dense1_weights.get_shape().as_list()[0], -1])
				print(fc1.get_shape())

				fc2 = tf.reshape(conv4a, [self.hy_dense2_weights.get_shape().as_list()[0], -1])
				print(fc2.get_shape())

				fc3 = tf.reshape(conv4a, [self.hy_dense3_weights.get_shape().as_list()[0], -1])
				print(fc3.get_shape())

				fc4 = tf.reshape(conv4a, [self.hy_dense4_weights.get_shape().as_list()[0], -1])
				print(fc4.get_shape())


				return fc1, fc2, fc3, fc4

			
		self.fc1,self.fc2,self.fc3,self.fc4 = model(self.tf_input_vector)

	def param(self, inputs, ground_truths, class_labels,  bbox):
		print("--", self.fc2.shape)

		fc1 = self.fc1
		fc2 = self.fc2
		fc3 = self.fc3
		fc4 = self.fc4

		#bbox = np.reshape(bbox, newshape = (4, -1))
		#print(bbox[0][0],bbox[1][0],bbox[2][0],bbox[3][0],bbox)
		#sys.exit()
		bbox0 = np.reshape(bbox[0], newshape = (1, -1))		
		bbox1 = np.reshape(bbox[1], newshape = (1, -1))		
		bbox2 = np.reshape(bbox[2], newshape = (1, -1))		
		bbox3 = np.reshape(bbox[3], newshape = (1, -1))		


		feed_dict = {self.tf_input_vector : inputs}

		#cls_loss = self.class_loss(class_labels, fc1)
		bx_loss_fc1 = self.bbox_loss(bbox0, self.hy_dense1_weights)
		bx_loss_fc2 = self.bbox_loss(bbox1, self.hy_dense2_weights)
		bx_loss_fc3 = self.bbox_loss(bbox2, self.hy_dense3_weights)
		bx_loss_fc4 = self.bbox_loss(bbox3, self.hy_dense4_weights)

		optimizer_1 = self.optimize(bx_loss_fc1)
		optimizer_2 = self.optimize(bx_loss_fc2)
		optimizer_3 = self.optimize(bx_loss_fc3)
		optimizer_4 = self.optimize(bx_loss_fc4)

		for epoch in range(0, 1000000):		
								
			_, _, _, _, loss1, loss2, loss3, loss4 = self._sess.run([optimizer_1, optimizer_2, optimizer_3, optimizer_4, bx_loss_fc1, bx_loss_fc2, bx_loss_fc3, bx_loss_fc4], feed_dict = feed_dict)
			fc1_, fc2_, fc3_, fc4_ = self._sess.run([self.hy_dense1_weights, self.hy_dense2_weights, self.hy_dense3_weights ,self.hy_dense4_weights])	#, feed_dict = feed_dict)
			print("epoch ", epoch, " BBOX Loss ", loss1, loss2, loss3, loss4 ," values :" , fc1_, fc2_, fc3_, fc4_, " box ", bbox[0], bbox[1], bbox[2], bbox[3])
		
		fig, ax = plt.subplots(1)
		ax.imshow(I)	
		rect = patches.Rectangle((fc2_[0], fc2_[1]), fc2_[2], fc2_[3], linewidth = 1, edgecolor = 'r')	
		ax.add_patch(rect)
		plt.show()

	def bbox_loss(self, bbox, prediction):
		print(self.fc2_output , prediction.shape, bbox.shape)
		loss = tf.losses.mean_squared_error(labels = bbox, predictions = prediction)
		loss += 0.001
		return loss

	def optimize(self, l2):


		global_step = tf.Variable(0)
		learning_rate	= 0.0001
		
		optimizer_2 = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999).minimize(l2, global_step = global_step)

		init = tf.global_variables_initializer()	
		self._sess.run(init)
		return optimizer_2
		







	



				
			



