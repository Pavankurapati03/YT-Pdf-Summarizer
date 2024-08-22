import pyttsx3

def speak_text(summary_text):
    engine = pyttsx3.init()
    engine.say(summary_text)
    engine.runAndWait()
