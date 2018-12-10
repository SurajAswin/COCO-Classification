import numpy as np
import cv2
import csv
import glob, os, sys

car_labels = "./ground_truth/car/*.jpg"
trafficlight_labels = "./ground_truth/traffic_lights/*.jpg"

def create_csv():
	count = 0
	print("creating csv file...")
	
	with open('classfile.csv', 'w') as csvfile:
		for filename in glob.glob(car_labels):
			filewriter = csv.writer(csvfile, delimiter = ",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
			basename = os.path.basename(filename)
			print(basename)
			if count == 0:
				filewriter.writerow(["FileName", "Class"])
			count = count + 1
			filewriter.writerow([basename, 0])	
		
	with open('classfile.csv', 'a') as file_:
		print(trafficlight_labels)
		for filename in glob.glob(trafficlight_labels):
			print("Now writing")
			filewriter = csv.writer(file_, delimiter = ",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
			basename = os.path.basename(filename)
			print(basename)
			filewriter.writerow([basename, 1])
	
	print("completed creating classfile")	

create_csv()
