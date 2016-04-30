from collections import deque
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()

ap.add_argument("-b", "--buffer", type=int, default=1000,
	help="max buffer size") #this is used to draw the trail behind the object.
args = vars(ap.parse_args())


greenLower = (29, 86, 6)
greenUpper = (80, 255, 255)

blueLower = (110,50,50)
blueUpper = (130, 255 , 255)

pts = deque(maxlen=args["buffer"]) #stores the set of points the object moves on.
new = deque(maxlen=args["buffer"])

camera = cv2.VideoCapture(0)

while True:
	
	(grabbed, frame) = camera.read() #take the available frame
    
	frame = imutils.resize(frame, width=600) # reduce the frame size to increase FPS.
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct masks for the colors , one mask per color.

    
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	mask2 = cv2.inRange(hsv, blueLower, blueUpper)
	mask2 = cv2.erode(mask2, None, iterations=2)
	mask2 = cv2.dilate(mask2, None, iterations=2)    

	# find contours in the masks and initialize the current (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
	cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
        
	center2 = None
	# check if countours exist.
	if len(cnts) > 0 :
		# Find largest countour , based on that trace the centroid.
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# Set threshold for radius
		if radius > 10:
			# draw the circle and centroid on the frame, then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	    
	if len(cnts2) > 0 :

		k = max(cnts2, key=cv2.contourArea)
		((x2, y2), radius2) = cv2.minEnclosingCircle(k)
		N = cv2.moments(k)
		center2 = (int(N["m10"] / N["m00"]), int(N["m01"] / N["m00"]))

		# only proceed if the radius meets a minimum size
		if radius2 > 10:


			cv2.circle(frame, (int(x2), int(y2)), int(radius2),
				(0, 255, 255), 2)
			cv2.circle(frame, center2, 5, (0, 0, 255), -1)

	# update the points queue
	new.appendleft(center2)
    
	# loop over the points tracked
	for i in xrange(1, len(pts)):

		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1)
		cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)
	
	for i in xrange(1, len(new)):
		# if either of the tracked points are None, ignore
		# them
		if new[i - 1] is None or new[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness2 = int(np.sqrt(args["buffer"] / float(i + 1)) * 1)
		cv2.line(frame, new[i - 1], new[i], (255, 0, 0), thickness2)
        
        
	cv2.imshow("Frame", frame)
	cv2.imshow("Green", mask)
	cv2.imshow("Blue", mask2)    
	key = cv2.waitKey(1) & 0xFF

	# press e to terminate
	if key == ord("e"):
		break
        


# cleanup functions
camera.release()
cv2.destroyAllWindows()
