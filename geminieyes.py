from dotenv import load_dotenv
from PIL import Image
import os
import google.generativeai as genai

import cv2

load_dotenv()

import datetime
import speech_recognition as sr
import pyttsx3

r=sr.Recognizer()
engine=pyttsx3.init()


genai.configure(api_key="AIzaSyDHtcnxpqeXI6EHZOgFtFiLWZIPx4KBUPQ")
model = genai.GenerativeModel('gemini-pro-vision')



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
        cv2.imshow("Live camera",frame)

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

# Example usage:
def text():
    with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                print("Ask your query")
                audio=r.listen(source)
                Query = r.recognize_google(audio)
                print("You said: ",Query)
    return Query
    
txt=text()
print(txt)
image_path = "D:/Projects/screenshots/screenshot_1"
 # Set to None if no image

#content=[input_text, image_path]
response = model.generate_content(txt,image_path,stream=True)
print("The Response is:")
print(response)


