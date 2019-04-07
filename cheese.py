import cv2
import os

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

cv2.imwrite('captured.jpg', frame)
os.system("gpup captured.jpg")