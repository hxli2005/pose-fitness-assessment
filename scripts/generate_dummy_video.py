import cv2
import numpy as np

print(">>> Generating dummy video (data/test_dummy.mp4)...")
fps = 30
width, height = 640, 480
out = cv2.VideoWriter('data/test_dummy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Generate 30 frames (1 second) of a moving block
for i in range(30):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200 # Light gray background

    # A moving dark grey block
    y_pos = 100 + i * 5
    cv2.rectangle(frame, (200, y_pos), (400, y_pos+200), (50, 50, 50), -1)

    out.write(frame)

out.release()
print(">>> Created 30-frame dummy video.")
