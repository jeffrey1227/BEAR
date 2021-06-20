import cv2
import numpy as np
import os
import math
import time

from aruco import findArucoMarkers
from mediapipe_hand import detectHandPose
from ar import *


from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
from objloader import *
from imutils.video import VideoStream
import cv2.aruco as aruco
import yaml
import imutils

import ctypes
import os

from pyglet.gl import *
from pywavefront import visualization, Wavefront

window = pyglet.window.Window(width=1080, height=720, resizable=True)

root_path = os.path.dirname(__file__)

bball = Wavefront(os.path.join(root_path, 'data/basketball-lowpoly/basketBall_OBJ.obj'), collect_faces=True)

# intrinsic param
K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])


object_dict = {'fox': ['../3d_objects/fox/fox.obj', '../3d_objects/fox/texture.png'],
               'basketball-wilson': ['../3d_objects/basketball-wilson/basketball-wilson.obj', '../3d_objects/basketball-wilson/basketball-wilson.jpg'],
               'basketball': ['../3d_objects/basketball/basketball.obj', '../3d_objects/basketball/basketball.jpg'],
               'basketball-molten': ['../3d_objects/basketball-molten/molten.obj', '../3d_objects/basketball-molten/molten.jpg'],
               'board': ['../3d_objects/board/board.obj', '../3d_objects/board/board.jpg']}

obj = ThreeDimObject(object_dict['basketball-wilson'][0], object_dict['basketball-wilson'][1])

def main():
    INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        start_time = time.time()
        
        success, image = cap.read()
        if not success:
            continue
        
        
        dst_pts, ids = findArucoMarkers(image)
        # if ids is not None and dst_pts is not None: 
        #     rvecs, tvecs, _objpoints = aruco.estimatePoseSingleMarkers(dst_pts[0], 0.6, K)
        #     #build view matrix
        #     # board = aruco.GridBoard_create(6,8,0.05,0.01,aruco_dict)
        #     # corners, ids, rejectedImgPoints,rec_idx = aruco.refineDetectedMarkers(gray,board,corners,ids,rejectedImgPoints)
        #     # ret,rvecs,tvecs = aruco.estimatePoseBoard(corners,ids,board,self.cam_matrix,self.dist_coefs)
        #     rmtx = cv2.Rodrigues(rvecs)[0]

        #     view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0][0][0]],
        #                             [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[0][0][1]],
        #                             [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[0][0][2]],
        #                             [0.0       ,0.0       ,0.0       ,1.0    ]])

            # view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
            #                         [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
            #                         [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
            #                         [0.0       ,0.0       ,0.0       ,1.0    ]])

            # view_matrix = view_matrix * INVERSE_MATRIX

            # view_matrix = np.transpose(view_matrix)

            # load view matrix and draw shape
            # glPushMatrix()
            # glLoadMatrixd(view_matrix)

            # glCallList(self.bball.gl_list)

            # glPopMatrix()
        
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