#!/usr/bin/env python
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import os, cv2
import numpy as np

## USAGE
# ./label_parser /path/to/imgs
## OUTPUT IMAGE ## 
# 8 bit grayscale 
# 0 = background
# 127 = door
# 255 = doorframe

def label(xml):
	tree = ET.parse(xml)
	doors = []
	doorframes = []
	for obj in tree.findall('object'):
		cnt = []
		for pt in obj.findall('polygon/pt'):
			x = int(pt.find('x').text)
			y = int(pt.find('y').text)
			cnt += [[x, y]]
		name = obj.find('name').text
		if name == 'door':
			doors += [np.array(cnt, dtype=np.int32)]
		elif name == 'doorframe':
			doorframes += [np.array(cnt, dtype=np.int32)]
	return doors, doorframes

if __name__ == '__main__':
	try:
		path = sys.argv[1]
	except:
		path = os.getcwd();
	for filename in os.listdir(path):
		if not filename.endswith('.xml'): continue
		fullname = os.path.join(path, filename)
		root, ext = os.path.splitext(filename)
		original = cv2.imread(root+'.jpg')
		height, width, depth = original.shape
		img = np.zeros([height, width], np.uint8)
		doors, doorframes = label(fullname)
		for door in doors:
			cv2.drawContours(img, [door], 0, (127, 255, 255), -1)
		for df in doorframes:
			cv2.drawContours(img, [df], 0, (255, 255, 255), -1)
		
		## Uncomment to view images
		# cv2.imshow('original', original)
		# cv2.imshow('labels', img)
		# cv2.waitKey(3000)
		# cv2.destroyAllWindows()
		cv2.imwrite(root+'_label.jpg', img)