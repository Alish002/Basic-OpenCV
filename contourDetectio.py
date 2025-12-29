# in the name of God
# contour detction

import cv2
import numpy as np



image = cv2.imread('cirb.tif')
assert image is not None, 'iamge not found !'

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur
blur = cv2.GaussianBlur(img, (7,7), 0)

#threshold
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((9,9), np.uint8)

# my opening
erosion = cv2.erode(thresh, kernel, iterations=2)
dilation = cv2.dilate(erosion, kernel, iterations=1)

# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# detection
contours, hierarchy= cv2.findContours(image=dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
print(f'contouts:{len(contours)}')

imgCopy = img.copy()
cv2.drawContours(image=imgCopy, contours=contours, contourIdx=-1, color=(0,255,0),thickness=2, lineType=cv2.LINE_AA)

cv2.imshow('erosion', erosion)
cv2.imshow('dilation', dilation)
cv2.imshow('binary image', thresh)
cv2.imshow('imgCopy', imgCopy)

cv2.waitKey(0)
cv2.destroyAllWindows()