from ultralytics import YOLO
import cv2
import os

# Initialize YOLO model
model = YOLO("yolov8x.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Create a directory to save screenshots
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Variable to keep track of screenshot count
screenshot_count = 0

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Make predictions on the frame
        results = model.predict(frame, show=True)

        # Display predictions
        print(results)

        # Check if 's' key is pressed to save a screenshot
        key = cv2.waitKey(1)
        if key == ord('s'):
            screenshot_count += 1
            screenshot_filename = os.path.join(screenshot_dir, f"screenshot_{screenshot_count}.png")
            cv2.imwrite(screenshot_filename, frame)
            print(f"Screenshot saved: {screenshot_filename}")

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break

finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
