import cv2
from config import STREAMS_CONFIG

stream_config = STREAMS_CONFIG[15]
roi = stream_config["rois"]["DOOR_ROI"]

rx = int(roi[0, 0])
ry = int(roi[0, 1])
rw = int(roi[2, 0] - rx)
rh = int(roi[2, 1] - ry)

cap = cv2.VideoCapture("GF-23-14-M.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 2990)
ret, frame = cap.read()
if ret:
    cv2.imwrite("scratch/door_frame_2990.png", frame)
    print("Saved scratch/door_frame_2990.png")
cap.release()
