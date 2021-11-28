try:
    from cv2 import cv2
except ImportError:
    pass
import numpy as np

image = cv2.imread("image/boule2.jpg")

twoDimage = image.reshape((-1, 3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
attempts = 10

ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape(image.shape)

cv2.imshow("k", result_image)

gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
edges = cv2.Canny(gray, 60, 120)
edges_filtered = cv2.Canny(gray_filtered, 60, 120)

cv2.imshow("seg", edges_filtered)

circles = cv2.HoughCircles(edges_filtered, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(image, (i[0],i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow("Output", image)
cv2.waitKey(0)