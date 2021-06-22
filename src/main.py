import cv2
import numpy as np
import os
import math
import time

# from aruco import findArucoMarkers
from mediapipe_hand import detectHandPose
import ctypes
import os
from ar import *
from pyglet.gl import *
from pywavefront import visualization, Wavefront
def main():

    # window = pyglet.window.Window(width=1280, height=720, resizable=True)

    # root_path = os.path.dirname(__file__)

    # box1 = Wavefront(os.path.join(root_path, 'uv_sphere.obj'))
    # ball_points = (box1.vertices)

    # ball = np.zeros((3, len(ball_points)))
    # for i in range(len(ball_points)):
    #     ball[:, i] = (np.array(ball_points[i]))
    object_dict = {'fox': ['../3d_objects/fox/fox.obj', '../3d_objects/fox/texture.png'],
               'basketball-wilson': ['../3d_objects/basketball-wilson/basketball-wilson.obj', '../3d_objects/basketball-wilson/basketball-wilson.jpg'],
               'basketball': ['../3d_objects/basketball/basketball.obj', '../3d_objects/basketball/basketball.jpg'],
               'basketball-molten': ['../3d_objects/basketball-molten/molten.obj', '../3d_objects/basketball-molten/molten.jpg'],
               'board': ['../3d_objects/board/board.obj', '../3d_objects/board/board.jpg']}

    obj = ThreeDimObject(object_dict['basketball-wilson'][0], object_dict['basketball-wilson'][1])




    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start_time = time.time()
        success, image = cap.read()
        if not success:
            continue
        
        
        # bboxs, ids = findArucoMarkers(image)
        image, multi_hand_landmarks = detectHandPose(image, obj)
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