import cv2
import mediapipe as mp
import time
import numpy as np
from numpy.linalg import inv
import copy
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



def detectHandPose(image, ball):
	with mp_hands.Hands(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as hands:

		K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])

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

				print(f'Real Zoot\n{real_hand}')
				image_ball = copy.deepcopy(ball)
				

				points = []
				for idx in ([0, 4, 8, 12, 20]):
					xn = hand_landmarks.landmark[idx].x
					yn = hand_landmarks.landmark[idx].y
					zn = hand_landmarks.landmark[idx].z

					real_hand = np.array([[xn*x_rel], [yn*y_rel], [1]])
					real_hand = (zn + z_root) * scale * real_hand
					real_hand = np.dot(inv(K), real_hand)

					points.append(real_hand)
				points = np.array(points)

				R = 3
				dis_range = R
				_min_dis = float('inf')
				solved_value = [0, 0, 0]
				for _x in range(int(min(points[:, 0, 0]) - dis_range), int(max(points[:, 0, 0])+dis_range)):
					for _y in range(int(min(points[:, 1, 0]) - dis_range), int(max(points[:, 1, 0])+dis_range)):
						for _z in range(int(min(points[:, 2, 0]) - dis_range), int(max(points[:, 2, 0])+dis_range)):
							_sum = 0
							for i in range(points.shape[0]):
								_sum += (points[i][0][0] - _x)**2+(points[i][1][0] - _y)**2+(points[i][2][0] - _z)**2-R**2
		
							if abs(_sum) < _min_dis:
								solved_value[0] = _x 
								solved_value[1] = _y 
								solved_value[2] = _z
								_min_dis = abs(_sum)

				
				image_ball[0, :] += solved_value[0]
				image_ball[1, :] += solved_value[1]
				image_ball[2, :] += solved_value[2]

				# # x = Symbol('x')
				# # y =  Symbol('y')
				# # z = Symbol('z')
				# # solved_value=solve([(points[0][0][0] - x)**2+(points[0][1][0] - y)**2+(points[0][2][0] - z)**2-R**2,
				# # 					(points[1][0][0] - x)**2+(points[1][1][0] - y)**2+(points[1][2][0] - z)**2-R**2,
				# # 					(points[2][0][0] - x)**2+(points[2][1][0] - y)**2+(points[2][2][0] - z)**2-R**2,], [x,y,z])
				solved_value = np.array(solved_value).reshape((3, 1))
				solved_value = np.dot(K, solved_value)
				solved_value = solved_value / solved_value[2][0]
				# print(f'Scale\n{solved_value}')

				for x in range(-5, 5):
					for y in range(-5, 5):
						try:
							image[int(solved_value[1])+x, int(solved_value[0])+y] = 0
						except:
							continue

				image_ball = np.dot(K, image_ball)
				image_ball = image_ball / image_ball[2][0]
				for i in range(image_ball.shape[1]):
					try:
						image[int(image_ball[1][i]), int(image_ball[0][i])] = 0
					except:
						continue


		return image, results.multi_hand_landmarks	
	

def main():
	# For webcam input:
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
	with mp_hands.Hands(
		min_detection_confidence=0.7,
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