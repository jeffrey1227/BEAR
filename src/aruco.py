import cv2
import cv2.aruco as aruco
import numpy as np


def findArucoMarkers(img, draw=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_param)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            continue
        
        bboxs, ids = findArucoMarkers(image)        
        
        cv2.imshow('Image', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()