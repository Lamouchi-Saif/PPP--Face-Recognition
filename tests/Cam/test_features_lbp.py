import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# LBP parameters
radius = 1
n_points = 8 * radius
method = "uniform"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (320, 240))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method)

    # Normalize for display
    lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    lbp = lbp.astype(np.uint8)

    # Convert to BGR for stacking
    lbp_color = cv2.cvtColor(lbp, cv2.COLOR_GRAY2BGR)

    # Side-by-side view
    combined = np.hstack((frame, lbp_color))

    cv2.imshow("Original vs LBP", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()