import os
import sys
import time
import numpy as np
import _pickle as cPickle
from itertools import zip_longest
from dataset.COCO.CheckPointFiles import *
import sys
from PythonAPI.pycocotools import coco
from PythonAPI.pycocotools import *
import PIL.Image
import scipy.misc
from matplotlib import *
from PythonAPI.pycocotools import mask
from matplotlib import path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
from pylab import *
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


vis = 0
if not os.path.exists('./dataset/COCO/mask2014'):
	os.makedirs('./dataset/COCO/mask2014')

annTypes = ('instances','captions','person_keypoints')
annotations_type = annTypes[0]



datatype = 'train2014'
ann_file_ = './dataset/COCO/annotations/%s_%s.json' % (annotations_type,datatype)
#ann_file = np.load(ann_file_)


my_coco = coco.COCO(ann_file_)
my_ann = my_coco.anns

totalCatIds = my_coco.getCatIds(catNms = ['car']) # or traffic light
totalImgIds = my_coco.getImgIds(catIds=totalCatIds)
print(len(totalImgIds))
#sys.exit()

for p in range(len(totalImgIds)):
	catIds = my_coco.getCatIds(catNms = ['car'])
	imgIds = my_coco.getImgIds(catIds=catIds)
	img = my_coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
	I = cv2.imread("./dataset/COCO/images/train2014/" + img["file_name"])

	print(img["file_name"])
	save = img["file_name"]	

	annIds = my_coco.getAnnIds(imgIds=img["id"], catIds = catIds, iscrowd = None)
	anns = my_coco.loadAnns(annIds)
	my_coco.showAnns(anns)
	mask = my_coco.annToMask(anns[0])
	for i in range(len(anns)):
		mask += my_coco.annToMask(anns[i])
	plt.imsave("./ground_truth/car/"+str(save), mask)
	plt.imsave("./Real/car/"+str(save), I)


				

												
				



