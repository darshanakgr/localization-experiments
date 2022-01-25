import cv2
import numpy as np

matcher = cv2.BFMatcher(cv2.NORM_L2)

p1 = np.array([
    [1, 1, 1], [2, 3, 3], [100, 50, 75], [225, 225, 225]
]).astype(np.float32)

p2 = np.array([
    [1, 2, 1], [2, 3, 3], [225, 150, 175], [111, 156, 225]
]).astype(np.float32)

# query, train
match = matcher.match(p1, p2)

for m in match:
    print(p1[m.queryIdx], p2[m.trainIdx], m.distance)
