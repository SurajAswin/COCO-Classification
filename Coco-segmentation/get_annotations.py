import os
import sys
import numpy as np
#from PythonAPI import *
from PythonAPI.pycocotools import *
from PythonAPI.pycocotools import coco
import _pickle as cPickle

if not os.path.exists('./dataset/COCO/CheckPointFiles'):
	os.makedirs('./dataset/COCO/CheckPointFiles')

annTypes = ('instances','captions','person_keypoints')
annotations_type = annTypes[2]

for mode in range(0,2):
	if mode == 1:
		datatype = 'val2014'
		ann_file_ = './dataset/COCO/annotations/%s_%s.json' % (annotations_type,datatype)
		#ann_file = np.load(ann_file_)

	else:
		datatype = 'train2014'
		ann_file_ = './dataset/COCO/annotations/%s_%s.json' % (annotations_type,datatype)
		#ann_file = np.load(ann_file_)


	my_coco = coco.COCO(ann_file_)
	my_ann = my_coco.anns

	prev_id = -1
	prev_count = 1  
	cnt = 0
	coco_kpt = my_ann

	for id in my_ann:

		current_id = my_ann[id]['image_id']
		if current_id == prev_id:
			prev_count = prev_count + 1
		else:
			prev_count = 1
			cnt = cnt + 1
		
		coco_kpt[id]['image_id'] = current_id
		coco_kpt[id]['bbox'] = my_ann[id]['bbox']
		coco_kpt[id]['segmentation'] = my_ann[id]['segmentation']
		coco_kpt[id]['area'] = my_ann[id]['area']
		coco_kpt[id]['id'] = my_ann[id]['id']
		coco_kpt[id]['iscrowd'] = my_ann[id]['iscrowd']
		coco_kpt[id]['keypoints'] = my_ann[id]['keypoints']
		coco_kpt[id]['num_keypoints'] = my_ann[id]['num_keypoints']
		width = my_coco.annToRLE(my_ann[id])
		coco_kpt[id]['img_width'] = width['size'][0]
		height = my_coco.annToRLE(my_ann[id])
		coco_kpt[id]['img_height'] = height['size'][1]
		print("W x H : ", coco_kpt[id]['img_width'], coco_kpt[id]['img_height'])
		prev_id = current_id
		print(' index , image_id : ', id , coco_kpt[id]['image_id'])
        	#print("Lenght of coco keypoints : ", len(coco_kpt))				

	if mode == 1:
		coco_val = coco_kpt
		f = open('./dataset/COCO/CheckPointFiles/coco_val.txt','wb')
		f.write(cPickle.dumps(coco_val))
		print('coco_val.txt generated')
	else: 
		f = open('./dataset/COCO/CheckPointFiles/coco_kpt.txt','wb')
		f.write(cPickle.dumps(coco_kpt))
		print('coco_kpt.txt generated')
	
				

