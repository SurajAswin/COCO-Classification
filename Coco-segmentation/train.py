import cv2
import numpy as np
import tensorflow as tf
import sys, os, glob, random, csv, operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import architecture


ground_path = "./ground_truth/input/"
input_path = "./Real/input/"

batch_size = 8

path, dirs, files = next(os.walk(str(input_path)))
Real_files = len(files)
print("Number of real images ", Real_files)
images = np.ndarray(shape = (1, 512, 512, 3), dtype = float)

path, dirs, files = next(os.walk(str(ground_path)))
Truth_files = len(files)
print("Number of GT images ", Truth_files)
segmented = np.ndarray(shape = (1, 512, 512, 3), dtype = float)

if Real_files == Truth_files:

	index = []
	truths = []
	real_files = os.listdir(input_path)

	for batch in range(0, 1):
		index.append(random.randrange(0, len(real_files)))

	for root, dirs, gt_files in os.walk(ground_path):
		for i in range(0, len(index)):

			if files[index[i]] in gt_files:
				truths.append(os.path.join(root, files[index[i]]))
	n = 0
	for image_id in range(len(index)):
		img = cv2.imread(str(input_path) + files[index[image_id]])
		res = cv2.resize(img, (512, 512))
		images[n,:,:,:] = res
		n += 1

	m = 0
	for gt_id in truths:
		img = cv2.imread(str(gt_id))
		res_gt = cv2.resize(img, (512, 512))
		segmented[m,:,:,:] = res_gt
		m += 1

	"""
	for i in range(8):
		cv2.imshow(str(files[index[i]]), images[i,:,:,:])
		cv2.waitKey(0)
		cv2.imshow(str(files[index[i]]), segmented[i,:,:,:])
		cv2.waitKey(0)
	
	"""

	bbox = []
	img_class = []

	with open("bbox_test.csv", 'rt', encoding='ascii') as f:
		reader = csv.reader(f)
		for row in reader:
			if row[0] == str(files[index[0]]):
				bbox.append(row[1])
				bbox.append(row[2])
				bbox.append(row[3])
				bbox.append(row[4])
				img_class.append(str(row[5]))

	print(str(files[index[0]]), "\n-----------\n", bbox, "\n-----------\n", img_class)

	if not bbox and not img_class:
		print("NO BBOX AND IMG CLASSES FOUND IN CSV")

	else:
		temp_box = set(bbox)
		temp_cls = set(img_class)
		final_box = []
		final_class = []

		for i in temp_box:
			final_box.append(float(i))
		
		for j in temp_cls:
			final_class.append(int(j))
	
		print("------------------------------------------------------------------------")
		
		print(final_box, len(final_box), final_class, len(final_class))

		print("------------------------------------------------------------------------")
		if len(final_box) == 4:
			with tf.Session() as sess:
				my_net = architecture(sess, 1, 512, 3, len(final_class), len(final_box))
				my_net.initialise(sess)
				my_net.param(images, segmented, final_class, np.asarray(final_box))
				sys.exit()
		else:
			print(len(final_box), " is more than or less than 4")	
			















