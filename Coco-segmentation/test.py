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
import csv
#plot
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

loading = ["car", "traffic light"]
mode = ["w", "a"]
sel = 0
nonimg = 0
for load in loading:

	totalCatIds = my_coco.getCatIds(catNms = [str(load)])
	totalImgIds = my_coco.getImgIds(catIds=totalCatIds)
	print(len(totalImgIds))
	#sys.exit()

	with open('bbox_test.csv', mode[sel]) as file_:

		#for filename in glob.glob(trafficlight_labels):
		for p in range(len(totalImgIds)):

			catIds = my_coco.getCatIds(catNms = [str(load)])
	
			imgIds = my_coco.getImgIds(catIds=catIds)
			img = my_coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
			
			for root, dirs, files in os.walk("./Real/input/"):
	
				if img["file_name"] in files:
				
					#img_from_coco = my_coco.getImgIds(imgIds = img["file_name"])
					annIds = my_coco.getAnnIds(imgIds=img["id"], catIds = catIds, iscrowd = None)
					anns = my_coco.loadAnns(annIds)
					my_coco.showAnns(anns)
					mask_ = my_coco.annToMask(anns[0])

					Rs = mask.encode(mask_)
					BBS = mask.toBbox(Rs)
					print(img["file_name"], img['id'],BBS)
					I = cv2.imread("./Real/input/"+img["file_name"])

					for box in (0, len(BBS), 4):
					
						fig, ax = plt.subplots(1)
						ax.imshow(I)	
						rect = patches.Rectangle((BBS[box], BBS[box+1]), BBS[box+2], BBS[box+3], linewidth = 1, edgecolor = 'r')	
						ax.add_patch(rect)
						plt.show()
		
					"""
					filewriter = csv.writer(file_, delimiter = ",", quotechar='|', 	quoting=csv.QUOTE_MINIMAL)
					filewriter.writerow([img["file_name"], BBS, str(load)])
				else:
					nonimg += 1
	sel = 1
					"""

print("NOT FOUND ", nonimg)
	

