# tests/create_test_video.py
import cv2
import numpy as np

def create_test_video(output_path: str, frames: int = 100, fps: int = 30):
    """Create a synthetic test video."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Draw some placeholders
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"Test video created: {output_path}")

if __name__ == "__main__":
    create_test_video("tests/test_video.mp4")
