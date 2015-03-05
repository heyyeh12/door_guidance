#!/usr/bin/env python
import sys
import cv2
import numpy as np

## TUNING PARAMETERS ##
dropped_frames = 1
hough_params = dict( rho = 1,
                     theta = np.pi/360,
                     threshold = 140,
                     srn = 0,
                     stn = 0
                     )


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
        lap_thresh = cv2.Laplacian(img_thresh,cv2.CV_8U)

        ## HOUGH TRANSFORM ##
        hough = cv2.HoughLines(lap_thresh, **hough_params)
        if hough is not None:
            for rho, theta in hough[0]:
                # only draws vertical lines
                if theta > np.pi/180*160 or theta < np.pi/180*20:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)


        ## DISPLAY IMAGES ##
        cv2.imshow('Thresh then Laplacian', lap_thresh)
        cv2.imshow('Image Threshold', img_thresh)
        cv2.imshow('Hough Lines', frame2)

        k = cv2.waitKey(30) & 0xff
        if k == 27: #escape key
            break
        elif k == ord('s'):
            cv2.imwrite('laplacian.png',lap)
            # cv2.imwrite('sobelx.png',sobx)
            # cv2.imwrite('sobely.png', soby)

        frame1 = frame2
        prev = next

    cap.release()
    cv2.destroyAllWindows()
