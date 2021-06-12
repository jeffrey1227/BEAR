import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detectHandPose(image):
	with mp_hands.Hands(
	min_detection_confidence=0.65,
	min_tracking_confidence=0.65) as hands:
		
		# print(image.shape) #(720, 1280, 3)

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

		return image, results.multi_hand_landmarks	
	

def main():
	# For webcam input:
	cap = cv2.VideoCapture(0)
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