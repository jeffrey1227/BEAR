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
    object_dict = {
               'basketball-wilson': ['../3d_objects/basketball-wilson/basketball-wilson.obj', '../3d_objects/basketball-wilson/basketball-wilson.jpg'],
               'basketball': ['../3d_objects/basketball/basketball.obj', '../3d_objects/basketball/basketball.jpg'],
               'basketball-molten': ['../3d_objects/basketball-molten/molten.obj', '../3d_objects/basketball-molten/molten.jpg'],
               'hoop': ['../3d_objects/board/hoop6.obj', '../3d_objects/board/board.jpg']}

    obj = ThreeDimObject(object_dict['basketball-wilson'][0], object_dict['basketball-wilson'][1])


    obj_name = 'hoop'

    obj_hoop = ThreeDimObject(object_dict[obj_name][0], object_dict[obj_name][1])

    dst_array = np.load('dst_array_full_hoop.npy')
    cap = cv2.VideoCapture(0)
    shooted = False
    solved_value = [0, 0, 0]
    t = 0
    while cap.isOpened():
        start_time = time.time()
        success, image = cap.read()
        if not success:
            continue
        
        # image = augment_v2(image, obj_hoop, dst_array)
        # bboxs, ids = findArucoMarkers(image)
        image, multi_hand_landmarks, shooted, solved_value, t = detectHandPose(image, obj, shooted, solved_value, t)
        # use landmarks to put ball on hand
        image = augment_v2(image, obj_hoop, dst_array)
        
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