from dotenv import load_dotenv
from PIL import Image
import os
import google.generativeai as genai
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

load_dotenv()

import datetime
import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
engine = pyttsx3.init()

genai.configure(api_key="AIzaSyDHtcnxpqeXI6EHZOgFtFiLWZIPx4KBUPQq")

model_yolo = YOLO("yolov8x.pt")
model_gemini = genai.GenerativeModel('gemini-pro-vision')

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

        # Make predictions on the frame using YOLO
        yolo_results = model_yolo.predict(frame, show=True)

        # Display YOLO predictions
        print(yolo_results)

        # Check if 's' key is pressed to save a screenshot
        key = cv2.waitKey(1)
        if key == ord('s'):
            screenshot_count += 1
            screenshot_filename = os.path.join(screenshot_dir, f"screenshot_{screenshot_count}.png")
            cv2.imwrite(screenshot_filename, frame)
            print(f"Screenshot saved: {screenshot_filename}")

            # Process the saved screenshot using Gem-Pro-Vision model
            gemini_response = model_gemini.generate_content([f"YOLO results: {yolo_results}", screenshot_filename])
            print("Gemini Response:")
            print(gemini_response.text)

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break

finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
