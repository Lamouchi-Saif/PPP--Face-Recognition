import cv2
import numpy as np
from skimage.feature import hog

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (320, 240))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract HOG features + visualization
    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    # Normalize HOG image for display
    hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
    hog_image = hog_image.astype(np.uint8)

    # Convert to BGR for display consistency
    hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR)

    # Stack original + HOG
    combined = np.hstack((frame, hog_image))

    cv2.imshow("Original vs HOG Features", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()