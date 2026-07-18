import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('paper.jpg')

img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

blured = cv2.GaussianBlur(img_gray, (9,9),0)

canny = cv2.Canny(blured, 100, 250)

contoutes, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contoutes))
print("-------\n",hierarchy)

cv2.drawContours(img, contoutes, -1 , (0,255,0),3)







cv2.imshow('org', img)
cv2.imshow('img_gray', img_gray)
cv2.imshow('blured', blured)
cv2.imshow('canny', canny)
cv2.imshow('draw', img)
cv2.imshow('contoutes and hierarchy', canny)
cv2.waitKey(0)