from dotenv import load_dotenv
load_dotenv()  # loading all the environment variables

import streamlit as st
import os
import google.generativeai as palm

import speech_recognition as sr
import pyttsx3


r=sr.Recognizer()
engine=pyttsx3.init()

# Configure Generative AI

palm.configure(api_key="AIzaSyDHtcnxpqeXI6EHZOgFtFiLWZIPx4KBUPQ")


# Initialize Generative Model




while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Ask your query")

        audio=r.listen(source)
        Query = r.recognize_google(audio)
        model = palm.generate_text(prompt=Query)
        print("You said: ",Query)

    
    
    reply = model.result   
    print(reply)
    engine.say(reply)
    engine.runAndWait()

