import cv2
import numpy as np
import os
import math
import time

from aruco import findArucoMarkers
from mediapipe_hand import detectHandPose
from ar import *

# intrinsic param
K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])


object_dict = {'fox': ['../3d_objects/fox/fox.obj', '../3d_objects/fox/texture.png'],
               'basketball': ['../3d_objects/basketball/basketball.obj', None],
               'board': ['../3d_objects/board/board.obj', '../3d_objects/board/board.jpg']}

obj = ThreeDimObject(object_dict['board'][0], object_dict['board'][1])

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start_time = time.time()
        
        success, image = cap.read()
        if not success:
            continue
        
        
        dst_pts, ids = findArucoMarkers(image)
        # if there is something, find homography
        if ids:
            # Calculate homography
            marker = cv2.imread('../markers/marker.png')
            marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
            H = calHomography(marker, dst_pts)

            R_T = get_extended_RT(K, H)
            transformation = K.dot(R_T) 
            
            image = augment(image, obj, transformation, marker, True)
            # image = np.flip(augment(image, obj, transformation, marker, True), axis = 1)

        # image, multi_hand_landmarks = detectHandPose(image)
        
        # use landmarks to put ball on hand
        
        
        # Calculate fps
        # end_time = time.time()
        # fps = 1 / (end_time - start_time)
        # start_time = end_time
        # cv2.putText(image, str(int(fps)) + ' fps', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv2.LINE_AA)
        
        
        cv2.imshow('Image', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break




if __name__ == '__main__':
    main()