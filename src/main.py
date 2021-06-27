import cv2
import numpy as np
import time

from mediapipe_hand import detectHandPose
from ar import *

K = np.load('../calibration/camera_parameters.npy', allow_pickle=True)[0]


def main():

    object_dict = {
               'basketball-wilson': ['../3d_objects/basketball-wilson/basketball-wilson.obj', '../3d_objects/basketball-wilson/basketball-wilson.jpg'],
               'hoop': ['../3d_objects/board/hoop6.obj', '../3d_objects/board/board.jpg']}

    obj = ThreeDimObject(object_dict['basketball-wilson'][0], object_dict['basketball-wilson'][1])
    obj_hoop = ThreeDimObject(object_dict['hoop'][0], object_dict['hoop'][1])
    
    cap = cv2.VideoCapture(0)

    try:
        dst_array = np.load('hoop_array.npy')
    except:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            marker = cv2.imread('../markers/marker.png')
            marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
            # dst_pts depends on the image shape
            # if ypur image differs from 1280x720, you can modify the numbers
            dst_pts = np.array([[[490., 180.],
                                [790., 180.],
                                [770., 450.],
                                [510., 450.]]])
            H = calHomography(marker, dst_pts)

            R_T = get_extended_RT(K, H)
            transformation = K.dot(R_T) 
            image, dst_array = save_hoop(image, obj_hoop, transformation, marker, True)
            break

    
    shot = False
    solved_value = [0, 0, 0]
    t = 0

    while cap.isOpened():
        start_time = time.time()
        success, image = cap.read()
        if not success:
            continue
        
        image, _, shot, solved_value, t = detectHandPose(image, obj, shot, solved_value, t)
        image = augment(image, obj_hoop, dst_array)
        
        #### Calculate fps #####
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        cv2.putText(image, str(int(fps)) + ' fps', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv2.LINE_AA)
        ########################
        
        cv2.imshow('Image', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break




if __name__ == '__main__':
    main()