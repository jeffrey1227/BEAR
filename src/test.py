import numpy as np

K = np.array([[982.16820326, 0., 534.53760602],
                [0., 984.53022526, 354.19493125],
                [0., 0., 1.]])


middle_finger_tip = np.array([[596.85, 27.70, 1.0]])
middle_finger_mcp = np.array([[554.68, 300.14, 1.0]])

middle_finger_tip = np.array([[625.01, 93.46, 1.0]])
middle_finger_mcp = np.array([[587.52, 296.65, 1.0]])

a = np.linalg.inv(K) @ middle_finger_tip.reshape(3, 1)

b = np.linalg.inv(K) @ middle_finger_mcp.reshape(3, 1)

dist = np.linalg.norm(a-b)
print(dist)