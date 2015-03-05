#!/usr/bin/env python
import sys
import cv2
import numpy as np
import math

## TUNING PARAMETERS ##
dropped_frames = 1
group_thresh = 60
hough_params = dict( rho = 1,
                     theta = np.pi/360,
                     threshold = 140,
                     srn = 0,
                     stn = 0
                     )
canny_params = dict( threshold1 = 200,
                     threshold2 = 250,
                     apertureSize = 5,
                     L2gradient = True )


class Cluster:
    def __init__(self):
        self.average_rho = 0.0
        self.average_theta = 0.0
        self.elements = []

    def distance(self, rho, theta):
        dist = math.fabs(theta - self.average_theta)
        if dist > math.pi:
            dist = math.pi*2 - dist

        dist_r = math.fabs(rho - self.average_rho)

        dist_c = math.sqrt(math.pow(dist,2) + math.pow(dist_r, 2))
        return dist_c

    def add_element(self, line):
        self.elements += [line]
        rho_sum = 0.0
        theta_sum = 0.0
        for e in self.elements:
            rho_sum += e[0]
            theta_sum += e[1]
        self.average_rho = rho_sum/len(self.elements)
        self.average_theta = theta_sum/len(self.elements)

    def get_average(self):
        return [self.average_rho, self.average_theta]

    def get_cluster_size(self):
        return len(self.elements)

def line_cluster(lines, img):
    clusters = []
    for line in lines:
        dist = []
        grouped = False
        for cluster in clusters:
            dist += [cluster.distance(line[0], line[1])]
        if len(dist) is not 0:
            if min(dist) < group_thresh:
                clusters[dist.index(min(dist))].add_element(line)
                grouped = True
        if not grouped:
            cluster = Cluster()
            cluster.add_element(line)
            clusters += [cluster]
    return clusters


if __name__ == '__main__':

    try:
        data = "door_data/"+sys.argv[1]+".mov"
    except:
        data = 0

    cap = cv2.VideoCapture(data)
    ret, frame1 = cap.read()
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    while(1):

        ## READ FRAME AND CONVERT TO GRAY SCALE ##
        for i in range(0, dropped_frames):
            ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(next,(7,7),0)
        ret,img_thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        ## LAPLACIAN FILTER ##
        edge_img = cv2.Laplacian(img_thresh,cv2.CV_8U)
        # edge_img = cv2.Canny(img_thresh, **canny_params)

        ## HOUGH TRANSFORM ##
        hough = cv2.HoughLines(edge_img, **hough_params)
        lines = []
        if hough is not None:
            for rho, theta in hough[0]:
                # only draws vertical lines
                if theta > np.pi/180*160 or theta < np.pi/180*20:
                    lines += [[rho,theta]]

        line_clusters = line_cluster(lines, frame2)

        for cluster in line_clusters:
            # print cluster.get_cluster_size()
            avg = cluster.get_average()
            a = np.cos(avg[1])
            b = np.sin(avg[1])
            x0 = a * avg[0]
            y0 = b * avg[0]
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)

        ## DISPLAY IMAGES ##
        cv2.imshow('Thresh then Edge', edge_img)
        cv2.imshow('Image Threshold', img_thresh)
        cv2.imshow('Hough Lines', frame2)

        k = cv2.waitKey(30) & 0xff
        if k == 27: #escape key
            break
        elif k == ord('s'):
            cv2.imwrite('laplacian.png',edge_img)
            # cv2.imwrite('sobelx.png',sobx)
            # cv2.imwrite('sobely.png', soby)

        frame1 = frame2
        prev = next

    cap.release()
    cv2.destroyAllWindows()
