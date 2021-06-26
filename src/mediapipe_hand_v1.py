import cv2
import mediapipe as mp
import time
import numpy as np
from numpy.linalg import inv
import copy

from numpy.linalg import inv
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from scipy import interpolate
import math 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


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

def detectHandPose(image, obj, shooted, solved_value, t):
	with mp_hands.Hands(
	min_detection_confidence=0.7,max_num_hands = 1,
	min_tracking_confidence=0.7) as hands:

		K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])

		vertices = obj.vertices
		color=True

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

		hoop = np.array([[640], [300], [1]])
		hoop = np.dot(inv(K), hoop)
		hoop = hoop * 20



		image = np.ascontiguousarray(image, dtype=np.uint8)

		if results.multi_hand_landmarks:
			# print(results.multi_hand_landmarks)
			# print('================================================================')
			for hand_landmarks in results.multi_hand_landmarks:
				if shooted == False:
					mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


					if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y :
						print('ball shooted')
						shooted = True


					x_rel = 1280
					y_rel = 720
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
					# print(f'Scale\n{solved_value}')

					# for x in range(-5, 5):
					# 	for y in range(-5, 5):
					# 		try:
					# 			image[int(two_d_solved_value[1])+x, int(two_d_solved_value[0])+y] = 0
					# 		except:
					# 			continue
					print('solved_value', solved_value)
					return image, results.multi_hand_landmarks, shooted, solved_value, t


		if shooted == True:


			R = 2
			npts = 10
			# p1= np.array(solved_value)
			# p2= hoop[:,0]
			p1 = solved_value #[2, 5, 84]
			print('ball starts to fly at', p1)
			p2 = [0, -5, 30.]

			theta = math.pi / 3
			g = 980

			sign = lambda a: (a>0) - (a<0)

			x_trans = p2[0] - p1[0]
			y_trans = p2[1] - p1[1]
			z_trans = p2[2] - p1[2]
			alpha = (x_trans**2 + z_trans**2)**(1/2)
			beta = alpha * math.tan(theta)

			theta_x = math.atan(beta/abs(x_trans))
			theta_z = math.atan(beta/abs(z_trans))			

			v0_z_pow = (z_trans * g) / math.sin(2*theta_z)
			v0_x_pow = (x_trans * g) / math.sin(2*theta_x)

			v0 = (abs(v0_x_pow) + abs(v0_z_pow))**(1/2)
			
			t_drop = 2 * v0 * math.sin(theta) / g
			dt = np.linspace(0, t_drop, npts)
			
			if v0_z_pow < 0:
				new_z = -abs(v0_z_pow)**(1/2) * math.cos(theta_z) * dt[t] + p1[2]
			else:
				print('')
				new_z = v0_z_pow**(1/2) * math.cos(theta_z) * dt[t] + p1[2]

			if v0_x_pow < 0:
				new_x = -abs(v0_x_pow)**(1/2) * math.cos(theta_x) * dt[t] + p1[0]
			else:
				new_x = v0_x_pow**(1/2) * math.cos(theta_x) * dt[t] + p1[0]
			
			
			new_y =  -(v0 * math.sin(theta) * dt[t] - 0.5 * g * (dt[t] **2)) + p1[1]
			
			shape3 = [new_x, new_y, new_z]
			print('ball position:', shape3) 

			# p1 = np.array([solved_value[2], solved_value[1], solved_value[0]])
			# p2 = np.array([hoop[2][0], hoop[1][0], hoop[0][0]])

			# npts = 20 # number of points to sample
			# y=np.array([0,-0.5,-1,-1,-0.5,0]) #describe your shape in 1d like this
			# amp=5
			# R = 2
			# z=np.arange(y.size)
			# xnew = np.linspace(z[0],z[-1] , npts) #sample the x coord
			# tck = interpolate.splrep(z,y,s=0) 
			# adder = interpolate.splev(xnew,tck,der=0)*amp
			# adder[0]=adder[-1]=0
			# adder=adder.reshape((-1,1))

			# #get a line between points
			# shape3=np.vstack([np.linspace(p1[dim],p2[dim],npts) for dim in range(3)]).T

			# #raise the z coordinate
			# shape3[:,-1]=shape3[:,-1]+adder[:,-1]

	
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
				shooted = False


		return image, results.multi_hand_landmarks, shooted, solved_value, t
	

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
				# print(results.multi_hand_landmarks)
				# print('================================================================')
				for hand_landmarks in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
					# print firger tip position
					print(f'WRIST\n{hand_landmarks.landmark[0]}')
					print(f'THUMB_TIP\n{hand_landmarks.landmark[4]}')
					print(f'INDEX_FINGER_TIP\n{hand_landmarks.landmark[8]}')
					print(f'MIDDLE_FINGER_TIP\n{hand_landmarks.landmark[12]}')
					print(f'RING_FINGER_TIP\n{hand_landmarks.landmark[16]}')
					print(f'PINKY_TIP\n{hand_landmarks.landmark[20]}')
			
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


'''
file_list = ['test_img/img1.jpg', 'test_img/img2.jpg', 'test_img/img3.jpg']
# For static images:
with mp_hands.Hands(
	static_image_mode=True,
	max_num_hands=2,
	min_detection_confidence=0.5) as hands:
	for idx, file in enumerate(file_list):
		# Read an image, flip it around y-axis for correct handedness output (see
		# above).
		image = cv2.flip(cv2.imread(file), 1)
		# Convert the BGR image to RGB before processing.
		results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

		# Print handedness and draw hand landmarks on the image.
		print('Handedness:', results.multi_handedness)
		if not results.multi_hand_landmarks:
			continue
		image_height, image_width, _ = image.shape
		annotated_image = image.copy()
		for hand_landmarks in results.multi_hand_landmarks:
			print('hand_landmarks:', hand_landmarks)
			print(
				f'Index finger tip coordinates: (',
				f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
				f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
			)
			mp_drawing.draw_landmarks(
				annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
		cv2.imwrite(
			'test_img/annotated_image' + str(idx+1) + '.png', cv2.flip(annotated_image, 1))

'''