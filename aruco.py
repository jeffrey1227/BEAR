import cv2
import cv2.aruco as aruco
import numpy as np

# intrinsic param
K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])


def findArucoMarkers(img, draw=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_param)
    print(ids)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids