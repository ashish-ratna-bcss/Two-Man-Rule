import cv2

cap = cv2.VideoCapture("GF-23-14-M.mp4")
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % 250 == 0:
        cv2.imwrite(f"scratch/milestone_{frame_idx}.png", frame)
        print(f"Saved scratch/milestone_{frame_idx}.png")
        
    frame_idx += 1
cap.release()
