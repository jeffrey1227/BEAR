import cv2
import numpy as np
import os
import math
import time

from aruco import findArucoMarkers
from mediapipe_hand import detectHandPose

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start_time = time.time()
        success, image = cap.read()
        if not success:
            continue
        
        
        bboxs, ids = findArucoMarkers(image)
        image, multi_hand_landmarks = detectHandPose(image)
        # use landmarks to put ball on hand
        
        
        # Calculate fps
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        cv2.putText(image, str(int(fps)) + ' fps', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv2.LINE_AA)
        
        
        cv2.imshow('Image', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break




if __name__ == '__main__':
    main()