#!/usr/bin/env python
import sys, os, errno
import time, datetime
import cv2

### USAGE ###
#
# ./data_capture [cam_port] [door_data/path/to/directory]
# cam_port - 0 for built in camera
#          - 1 for usb webcam
#
# esc      - close out
# space    - take picture and record distance
# s        - save picture w/o distance
#
#

def check_path(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

if __name__ == '__main__':
	try:
	    cam_port = int(sys.argv[1])
	except:
		cam_port = 0
	
	data_dir = 'door_data/'
	try:
		data_dir += sys.argv[2]
	except:
		data_dir += datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
	
	check_path(data_dir)
	os.chdir(data_dir)

	fo = open('distances.txt', "a")
	print "Data file path: ", data_dir+"/"+fo.name

	cam = cv2.VideoCapture(cam_port)

	i = 1
	while True:
		
		retval, img = cam.read()
		cv2.imshow("webcam", img)

		k = cv2.waitKey(30) & 0xff
		if k == 27: #escape key
			break
	
		elif k == 32: #space key
			img_name = 'pic{:>03}.jpg'.format(i)
			dist = raw_input("[" + str(i) + "] Distance (in meters): ")
			fo.write(img_name + "   " + str(dist) + "\n")
			cv2.imwrite(img_name, img)
			i+=1

		elif k == ord('s'):
			img_name = raw_input("Save as: ")
			cv2.imwrite((img_name+'.jpg').format(i), img)

	prev = next

	cam.release()
	cv2.destroyAllWindows()
	fo.close()