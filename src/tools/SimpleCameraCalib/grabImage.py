#!/usr/bin/python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
#
# opens camera device and displays resized original image and an edge filter
# with maximal feasible framerate. input is being parsed and can handle numeric
# device names as well as addresses. pressing s saves images.
#
# usage
#   grabImage.py -c <camera>
#
# example calls:
#   grabImage.py -c tcp://viscam1.hlrs.de:5001
#   grabImage.py -c 1
#
# -----------------------------------------------------------------------------

import numpy as np
import cv2
import sys
import getopt

img_size_x = 1920
img_size_y = 1080

# -----------------------------------------------------------------------------
def camTest(camName):

    if camName.isdigit():
        cap = cv2.VideoCapture(int(camName))
    else:
        cap = cv2.VideoCapture(camName)

    cap.set(3, img_size_x)
    cap.set(4, img_size_y)
        
    count = 0
    default_dir = './'

    while True:

        ret, frame = cap.read()
    
        frame_s = cv2.resize(frame, (640, 480)) 

        gray = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Canny(gray,100,200)

        cv2.imshow('orig',frame_s)
        cv2.imshow('edge',laplacian)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):

            break

        elif key == ord('s'):

            outfile = default_dir + 'snapshot_' + str(count).zfill(2) + '.png'
            count += 1
            print 'saving to ' + outfile
            cv2.imwrite(outfile, frame)

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
def main(argv):
    camName = ''

    try:
        opts, args = getopt.getopt(argv,"hc:",["camera="])
        
    except getopt.GetoptError:
        print 'grabImage.py -c <camera>'
        sys.exit(2)

    if opts == []:
        print 'using default camera'
        camName = '0'
    else:
        for opt, arg in opts:
            if opt == '-h':
                print 'grabImage.py -c <camera>'
                sys.exit(1)
            
            elif opt in ("-c","--camera"):
                camName = arg

    print 'camera = ' + camName

    camTest(camName)
    
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])
