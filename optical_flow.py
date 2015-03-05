#!/usr/bin/env python
import sys
import cv2
import numpy as np

## TUNING PARAMETERS ##
dropped_frames = 5
group_thresh = 50
threshold_params = dict ( thresh = 0,
                          maxval = 100,
                          type = cv2.THRESH_BINARY_INV)
optflow_params = dict( pyr_scale = 0.5,
                         levels = 1,
                         winsize = 5,
                         iterations = 15,
                         poly_n = 7, #or 5
                         poly_sigma = 1.5,
                         flags = 0) #or 1.1
canny_params = dict( threshold1 = 51,
                     threshold2 = 101,
                     apertureSize = 3,
                     L2gradient = True )
hough_params = dict( rho = 1,
                     theta = np.pi/180,
                     threshold = 123,
                     srn = 0,
                     stn = 0)

## DEFAULT OPTICAL FLOW VISUALIZATION ##
def draw_flow(img, flow, step=16):
    threshold = 10
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

## CUSTOMIZED FUNCTIONS ##
def draw_hsv_thresh(flow, params):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,2] = cv2.normalize(mag,None,0,100,cv2.NORM_MINMAX)
    h, s, v = cv2.split(hsv)
    # v = cv2.GaussianBlur(v, (5,5), 0)
    ret, new = cv2.threshold(v, **params)
    return new

def draw_hough(img, hough):
    lines = []
    if hough is not None:
            for rho, theta in hough[0]:
                # only draws near-vertical lines
                if theta > np.pi/180*165 or theta < np.pi/180*15:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    lines += [(x1+x2)/2]
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return lines


def line_cluster(lines, img):
    groups = []
    avgs = []
    for line in lines:
        diff = []
        grouped = False
        for group in groups:
            group_avg = sum(group)/len(group)
            diff += [abs(group_avg-line)]
        if len(diff) is not 0:
            if min(diff) < group_thresh:
                groups[diff.index(min(diff))] += [line]
                grouped = True
        if not grouped:
            groups += [[line]]
    for group in groups:
        if len(group) is not 0:
            avg = int(sum(group)/len(group))
            cv2.line(img, (avg, -1000), (avg, 1000), (0, 255, 0), 2)
            avgs += [avg]
    if len(avgs) is not 0:
        cv2.line(img, (int(sum(avgs)/len(avgs)), -1000), (int(sum(avgs)/len(avgs)), 1000), (255, 0, 0), 2)
    return groups

if __name__ == '__main__':

    try:
        data = "door_data/"+sys.argv[1]+".mov"
    except:
        data = 0

    cap = cv2.VideoCapture(data)
    ret, frame1 = cap.read()
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    while(1):
        
        ## OPTICAL FLOW ##
        for i in range(0, dropped_frames):
            ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, **optflow_params)
        vfield_img = draw_flow(next, flow)
        hsv1_img = draw_hsv(flow)
        hsv2_img = draw_hsv_thresh(flow, threshold_params)

        ## HOUGH FILTER ##
        canny = cv2.Canny(next, **canny_params)
        hough = cv2.HoughLines(canny, **hough_params)
        lines_hough = draw_hough(frame2, hough)
        groups_hough = line_cluster(lines_hough, frame2)
        # print groups_hough
        
        canny = cv2.Canny(hsv2_img, **canny_params)
        hough = cv2.HoughLines(canny, **hough_params)
        lines = draw_hough(hsv2_img, hough)
        groups = line_cluster(lines, hsv2_img)

        ## DISPLAY IMAGES ##
        # cv2.imshow('Optical Flow Field', vfield_img)
        # cv2.imshow('Optical Flow HSV', hsv1_img)
        cv2.imshow('Optical Flow HSV Threshold', hsv2_img)
        # cv2.imshow('Canny', canny)
        cv2.imshow('Hough Lines', frame2)

        k = cv2.waitKey(30) & 0xff
        if k == 27: #escape key
            break
        elif k == ord('s'):
            cv2.imwrite('optflow_field.png',frame2)
            cv2.imwrite('optflow_hsv.png',rgb)
        
        prev = next

    cap.release()
    cv2.destroyAllWindows()