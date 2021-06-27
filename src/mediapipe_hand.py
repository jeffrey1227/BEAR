import cv2
import mediapipe as mp
import time
import numpy as np
from numpy.linalg import inv
import copy

from numpy.linalg import inv
from numpy.linalg import norm
import math 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


K = np.load('../calibration/camera_parameters.npy', allow_pickle=True)[0]


def trilaterate(points, distances):

	p1,p2,p3 = points
	r1,r2,r3 = distances

	ex = (p2-p1) / norm(p2-p1)
	i = np.dot(ex, p3-p1)
	a = (p3-p1) - ex*i
	ey = a / norm(a)
	ez = np.cross(ex, ey)
	d = norm(p2-p1)
	j = np.dot(ey, p3-p1)
	x = (r1**2 - r2**2 + d**2) / (2*d)
	y = (r1**2 - r3**2 + i**2 + j**2) / (2*j) - (i/j) * x
	b = r1**2 - x**2 - y**2

	if (np.abs(b) < 0.0000000001):
		b = 0
	try:
		z = np.sqrt(b)
		if np.isnan(z):
			raise Exception('NaN met, cannot solve for z')
	except:
		return False, False, False

	a = p1 + ex*x + ey*y
	pa = a + ez*z
	pb = a - ez*z

	return pa, pb, True

def detectHandPose(image, obj, shot, solved_value, t):
	with mp_hands.Hands(
	min_detection_confidence=0.7,max_num_hands = 1,
	min_tracking_confidence=0.7) as hands:

		vertices = obj.vertices
		color=True
		h, w, _ = image.shape # 720, 1280

		# Flip the image horizontally for a later selfie-view display, and convert
		# the BGR image to RGB.
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		results = hands.process(image)

		# Draw the hand annotations on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		hoop = np.array([[320], [200], [1]])
		hoop = np.dot(inv(K), hoop)
		hoop = hoop * 20

		image = np.ascontiguousarray(image, dtype=np.uint8)

		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				if not shot:


					if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y :
						print('ball shot')
						shot = True

					# reconstruct 3D pose from 2.5D representation
					x_rel = w
					y_rel = h
					xn = hand_landmarks.landmark[5].x 
					xm = hand_landmarks.landmark[0].x 
					yn = hand_landmarks.landmark[5].y 
					ym = hand_landmarks.landmark[0].y
					zn = hand_landmarks.landmark[5].z
					zm = hand_landmarks.landmark[0].z
					C = 1
					a = (xn - xm)**2 + (yn - ym)**2
					b = zn * (xn**2+yn**2 - xn*xm - yn*ym) + zm*(xm**2+ym**2-xn*xm-yn*ym)
					c = (xn*zn - xm*zm)**2 + (yn*zn - ym*zm)**2 + (zn-zm) ** 2 - C**2
					z_root = 0.5*(-b + (b**2 - 4*a*c)**1/2) / a


					scale = 9 / (((xn - xm) ** 2 + (yn - ym) ** 2 + (zn - zm) ** 2)**(1/2))
					
					real_hand = np.array([[xm*x_rel], [ym*y_rel], [1]])
					real_hand = (zm + z_root) * scale * real_hand
					real_hand = np.dot(inv(K), real_hand)

					hand_z = real_hand[2,0]

					points = []
					for idx in [0, 4, 8, 12, 16, 20]:
						xn = hand_landmarks.landmark[idx].x
						yn = hand_landmarks.landmark[idx].y
						zn = hand_landmarks.landmark[idx].z

						real_hand = np.array([[xn*x_rel], [yn*y_rel], [1]])
						real_hand = (zn + z_root) * scale * real_hand
						real_hand = np.dot(inv(K), real_hand)

						points.append(real_hand)
					points = np.array(points)


					# search best centroid of ball
					R = 2
					dis_range = 5
					_min_dis = float('inf')
					solved_value = [0, 0, 0]
					for _x in range(int((points[0, 0, 0]) - dis_range), int((points[0, 0, 0])+dis_range)):
						for _y in range(int((points[0, 1, 0]) - dis_range), int((points[0, 1, 0])+dis_range)):
							for _z in range(int((points[0, 2, 0]) - dis_range*2), int((points[0, 2, 0])+dis_range*2)):
								_sum = 0
								for i in range(points.shape[0]):
									_sum += (points[i][0][0] - _x)**2+(points[i][1][0] - _y)**2+(points[i][2][0] - _z)**2-R**2
			
								if abs(_sum) < _min_dis:
									solved_value[0] = _x 
									solved_value[1] = _y 
									solved_value[2] = _z
									_min_dis = abs(_sum)

					# render ball in palm
					for face in obj.faces:
						#a face is a list [face_vertices, face_tex_coords, face_col]
						face_vertices = face[0]
						points = np.array([vertices[vertex - 1] for vertex in face_vertices]) #-1 because of the shifted numbering
						points = R * points
						points = np.array([[p[2] + solved_value[0], p[0] + solved_value[1], p[1] + solved_value[2]] for p in points]) #shifted to centre 

						points = np.dot(K, points.T)
						dst = points / points[2, :]

						dst = dst.T
						dst = dst[:, :2]
						dst = dst.reshape(-1, 1, 2)

						imgpts = np.int32(dst)
						if color is False:
							cv2.fillConvexPoly(image, imgpts, (50, 50, 50))
						else:
							cv2.fillConvexPoly(image, imgpts, face[-1])


					two_d_solved_value = np.array(solved_value).reshape((3, 1))
					two_d_solved_value = np.dot(K, two_d_solved_value)
					two_d_solved_value = two_d_solved_value / two_d_solved_value[2][0]
					return image, results.multi_hand_landmarks, shot, solved_value, t

		# ball shot, calculate and render trajectory
		if shot:

			R = 2
			npts = 10
			p1= np.array(solved_value)
			print('ball starts to fly at', p1)
			p2 = [0, -6, 20.]
			
			theta = math.pi / 3
			g = 980

			x_trans = p2[0] - p1[0]
			y_trans = p2[1] - p1[1]
			z_trans = p2[2] - p1[2]
			alpha = (x_trans**2 + z_trans**2)**(1/2)

			v0 = ((alpha * g) / math.sin(2*theta)) ** (1/2)

			percet = abs(z_trans) / abs(x_trans)
			v0_z = - v0 / (math.sqrt(percet ** 2 + 1)) * percet
			v0_x = (v0**2 - v0_z**2)**(1/2) * np.sign(x_trans)


			t_drop = 2 * v0 * math.sin(theta) / g
			dt = np.linspace(0, t_drop, npts)

			new_z = v0_z* math.cos(theta) * dt[t] + p1[2]
			new_x = v0_x * math.cos(theta) * dt[t] + p1[0]		
			new_y =  -(v0 * math.sin(theta) * dt[t] - 0.5 * g * (dt[t] **2)) + p1[1]
			
			shape3 = [int(new_x), int(new_y), int(new_z)]
			
			# render
			for face in obj.faces:
				#a face is a list [face_vertices, face_tex_coords, face_col]
				face_vertices = face[0]
				points = np.array([vertices[vertex - 1] for vertex in face_vertices]) #-1 because of the shifted numbering
				points = R * points
				points = np.array([[p[2] + shape3[0], p[0] + shape3[1], p[1] + shape3[2]] for p in points]) #shifted to centre 


				points = np.dot(K, points.T)
				dst = points / points[2, :]

				dst = dst.T
				dst = dst[:, :2]
				dst = dst.reshape(-1, 1, 2)
				imgpts = np.int32(dst)
				if color is False:
					cv2.fillConvexPoly(image, imgpts, (50, 50, 50))
				else:
					cv2.fillConvexPoly(image, imgpts, face[-1])

			t += 1
			if t >= npts:
				t = 0
				shot = False


		return image, results.multi_hand_landmarks, shot, solved_value, t
	

def main():
	# For webcam input:
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
	with mp_hands.Hands(
		min_detection_confidence=0.7, max_num_hands = 1,
		min_tracking_confidence=0.7) as hands:
		start_time = time.time()
		while cap.isOpened():
			
			success, image = cap.read()
			# print(image.shape) #(720, 1280, 3)
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue

			# Flip the image horizontally for a later selfie-view display, and convert
			# the BGR image to RGB.
			image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			results = hands.process(image)

			# Draw the hand annotations on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			if results.multi_hand_landmarks:
				for hand_landmarks in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
			
			# Calculate fps
			end_time = time.time()
			fps = 1 / (end_time - start_time)
			start_time = end_time
			cv2.putText(image, str(int(fps)) + ' fps', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv2.LINE_AA)	
			

			cv2.imshow('MediaPipe Hands', image)
			if cv2.waitKey(5) & 0xFF == 27:
				break

	cap.release()

if __name__ == '__main__':
	main()
