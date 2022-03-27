from __future__ import print_function
from Capture import FPS
from Capture import WebcamVideoStream
import imutils
import cv2
import argparse

stream = cv2.VideoCapture(0)
fps = FPS().start()

while fps._numFrames < 100:
    (grabbed, frame) = stream.read()
    frame = imutils.resize(frame, width =400)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    fps.update()
    
fps.stop()


print('[INFO] elasped time: {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

stream.release()
vs = WebcamVideoStream(src = 0).start()
fps = FPS().start()
while fps._numFrames < 100:
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    fps.update()

fps.stop()

print('[INFO] elasped time: {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

vs.stop()
cv2.destroyAllWindows()