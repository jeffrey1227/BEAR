import cv2
import numpy as np
import os
import math
import time

from aruco import findArucoMarkers
from mediapipe_hand import detectHandPose
from ar import *

from pywavefront import visualization, Wavefront


# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
# import cv2
# from PIL import Image
# import numpy as np
# from objloader import *
# from imutils.video import VideoStream
# import cv2.aruco as aruco
# import yaml
# import imutils

# import ctypes
# import os

# from pyglet.gl import *
# from pywavefront import visualization, Wavefront

# window = pyglet.window.Window(width=1080, height=720, resizable=True)

# root_path = os.path.dirname(__file__)

# bball = Wavefront(os.path.join(root_path, 'data/basketball-lowpoly/basketBall_OBJ.obj'), collect_faces=True)

# intrinsic param
K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])


object_dict = {'fox': ['../3d_objects/fox/fox.obj', '../3d_objects/fox/texture.png'],
               'basketball-wilson': ['../3d_objects/basketball-wilson/basketball-wilson.obj', '../3d_objects/basketball-wilson/basketball-wilson.jpg'],
               'basketball': ['../3d_objects/basketball/basketball.obj', '../3d_objects/basketball/basketball.jpg'],
               'basketball-molten': ['../3d_objects/basketball-molten/molten.obj', '../3d_objects/basketball-molten/molten.jpg'],
               'board2': ['../3d_objects/board/board2.obj', '../3d_objects/board/board.jpg'],
               'hoop6': ['../3d_objects/board/hoop6_reverse.obj', '../3d_objects/board/hoop6.png']}

obj_name = 'hoop6'

obj = ThreeDimObject(object_dict[obj_name][0], object_dict[obj_name][1])

def main():
    dst_array = np.load('dst_array.npy')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start_time = time.time()
        
        success, image = cap.read()
        if not success:
            continue
        
        
        # dst_pts, ids = findArucoMarkers(image)
        # print(dst_pts)
        # dst_pts = np.array([[[510., 180.],
        # [760., 180.],
        # [740., 400.],
        # [490., 400.]]])

        # ids = True
        
        # if there is something, find homography
        # if ids:
        #     # Calculate homography
        #     marker = cv2.imread('../markers/marker.png')
        #     marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
        #     H = calHomography(marker, dst_pts)

        #     R_T = get_extended_RT(K, H)
        #     transformation = K.dot(R_T) 
            
        image = augment_v2(image, obj, dst_array)
            # image = np.flip(augment(image, obj, transformation, marker, True), axis = 1)

        # image, multi_hand_landmarks = detectHandPose(image)
        
        
        
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