#!/usr/bin/env python2

import os,sys
import argparse
import numpy as np
import cv2


def main(prog_name,argv):
    PROP_FPS = 5
    
    parser = argparse.ArgumentParser( prog=prog_name, description='analyzes the fish',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument('-i','--infile', dest='infile', metavar='INFILE', type=str, required=True, help='video of the fishies')
    parser.add_argument('--display', dest='display', action='store_true', default=False, help='display one tenth of the video frames with a few markers')
    parser.add_argument('-o','--outfile', dest='outfile', metavar='OUTFILE', type=str, required=False, help='output file in csv format')
    args = parser.parse_args(argv)
    
    # INITIALIZE COMMAND LINE ARGS
    infile = args.infile
    outfile = args.outfile
    display = args.display

    # check for existence of file
    if not os.path.isfile( infile ):
        print 'Error: {0} not found'.format(infile)
        sys.exit(1)

    if outfile is None:
        outfile = os.path.splitext(infile)[0] + '.fish' 

    cap = cv2.VideoCapture(infile)
    fps = round(cap.get(PROP_FPS))
    #if fps != 30:
    #    print 'Error: FPS is: {0}'.format(fps)
    #    sys.exit()

    # INITIALIZE RUNNING VARIABLES
    lastframe = None
    dividers = [201,424]
    indicators = [(100,320),(310,320),(520,320)]
    sigtime = False
    sigcount = 0
    sigpos = [] 
    currentpos = 1
    AVI_RATIO = 2

    # read beginning of video to initialize loop
    if lastframe is None:
        # skip first 9 frames
        for i in xrange(10):
            ret,frame = cap.read()
            if cap.get(AVI_RATIO) == 1:
                sys.exit()
        
        b,g,r = cv2.split(frame)

        gray = cv2.mean(b,g)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find yellow dividers
    
    while(cap.isOpened()):
        # replace lastframe
        lastframe = gray

        # count 10 frames
        for i in xrange(10):
            ret,frame = cap.read()
            if cap.get(AVI_RATIO) == 1:
                break
        
        # PROCESS FRAME
        #  split frame into color channels
        #  find the mean of blue and green channels
        #  until found, look for the red light in the corner
        #  find the difference between this frame and the last frame processed
        b,g,r = cv2.split(frame)
        gray = cv2.mean(b,g)

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get difference 
        frame_diff = cv2.absdiff(gray, lastframe)
        ret,frame_diff = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        frame_diff = cv2.blur(frame_diff,(5,5))
        ret,frame_diff = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        frame_diff = cv2.blur(frame_diff,(20,20))
        ret,frame_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
        frame_diff = cv2.blur(frame_diff,(20,20))
        ret,frame_diff = cv2.threshold(frame_diff, 5, 255, cv2.THRESH_BINARY)

        # find contours in difference image
        cont,_ = cv2.findContours(frame_diff,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        # get copy of image to draw on
        gray_show = gray.copy()

        # draw green box around the biggest contour in the image
        if cont != []:
            fish = None
            for c in sorted(cont, key=lambda t: cv2.contourArea(t)):
                center,radius = cv2.minEnclosingCircle(c)
                if not sigtime and center[0] > 600 and center[1] > 310:
                    sigtime = True
                elif not (center[0] > 600 and center[1] > 310) and fish is None:
                    fishcenter = center
                    fishradius = radius
                    fish = c

            # check if display flag is present
            if display:
                cv2.circle( frame, (int(fishcenter[0]),int(fishcenter[1])), 5, (0,153,255), -1 )
                
            # set current position
            if fishcenter[0] < dividers[0]:
                currentpos = 0
            elif fishcenter[0] > dividers[1]:
                currentpos = 2
            else:
                currentpos = 1

        # draw indicator to show cell the fish is in    
        if sigtime and sigcount % 3 == 0:
            sigpos.append(currentpos)
            
            # check if display flag is present
            if display:
                cv2.circle( frame, indicators[currentpos], 5, (255,0,0), -1 )
            
        if sigtime:
            sigcount += 1

        if sigcount >= 120*3:
            break

        # if display flag is present
        if display:
            cv2.line( frame, (201,0), (201,1000), (0,255,0), 3 )
            cv2.line( frame, (424,0), (424,1000), (0,255,0), 3 )

            cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # convert sigpos members to str
    sigpos = [str(p) for p in sigpos]

    # print sigpos to file
    with open(outfile,'w') as fh:
        fh.write(','.join(sigpos) + '\n' )

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main(sys.argv[0],sys.argv[1:])
