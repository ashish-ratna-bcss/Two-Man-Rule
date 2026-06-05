import cv2
import numpy as np

cap = cv2.VideoCapture("GF-23-14-M.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Video Resolution:", width, "x", height, "FPS:", fps)
cap.release()
