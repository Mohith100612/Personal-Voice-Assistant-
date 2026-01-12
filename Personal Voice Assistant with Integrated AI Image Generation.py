from datetime import datetime
import speech_recognition as sr
import pyttsx3
import webbrowser
import wikipedia
import wolframalpha
import openai
import os
from dotenv import load_dotenv
import requests
import pywhatkit
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import matplotlib.pyplot as plt
from PIL import Image
import uuid

# Load environment variables
load_dotenv()

# --- TTS Setup ---
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

# --- Vosk Speech Recognition Setup ---
def init_vosk():
    vosk_model_path = r"E:\major\vosk-model-small-en-us-0.15"
    return Model(vosk_model_path)

# --- Stable Diffusion Setup ---
def init_stable_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_path = os.path.abspath(r"E:\major\stable-diffusion-local")

    if not os.path.exists(os.path.join(local_model_path, "model_index.json")):
        raise FileNotFoundError("Missing model_index.json in your local model folder!")

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=local_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device), device

# --- Initialize Services ---
tts_engine = init_tts()
vosk_model = init_vosk()
sd_pipeline, device = init_stable_diffusion()
wolfram_client = wolframalpha.Client('5R49J7-J888YX9J2V')

# --- Utility Functions ---
def speak(text):
    print(f"Assistant: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen(prompt_text="Speak now"):
    speak(prompt_text)
    print("Listening...")
    rec = KaldiRecognizer(vosk_model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                prompt = result.get("text", "").lower()
                print(f"You said: {prompt}")
                return prompt
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- Image Generation and Search ---
def generate_ai_image(prompt):
    speak(f"Generating AI image for: {prompt}")
    torch.cuda.empty_cache()
    image = sd_pipeline(prompt, width=768, height=768).images[0]
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    speak("AI-generated image displayed.")

def search_real_image(prompt):
    speak(f"Searching for: {prompt}")
    api_key = "5743a3073c8bcab7bceb28a79bdadb9bfee89224c61f80e47f8a77dd3d163aaf"
    params = {"q": prompt, "tbm": "isch", "api_key": api_key, "engine": "google"}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if not data.get("images_results"):
            speak("No images found.")
            return
        image_url = data["images_results"][0]["original"]
        print("Image URL:", image_url)
        headers = {"User-Agent": "Mozilla/5.0"}
        image_response = requests.get(image_url, headers=headers, stream=True)
        if image_response.status_code == 200 and "image" in image_response.headers.get("Content-Type", ""):
            try:
                image = Image.open(image_response.raw)
                plt.imshow(image)
                plt.axis("off")
                plt.show()
                speak("Real image displayed.")
            except Exception as e:
                print("Error displaying image:", e)
                speak("Failed to display image. Unsupported format.")
        else:
            speak("Invalid image URL.")
    except Exception as e:
        print("Error during image search:", e)
        speak("Image search failed.")

# --- Information Retrieval ---
def search_wikipedia(keyword):
    try:
        return wikipedia.summary(keyword, sentences=2)
    except:
        return "Wikipedia information not found."

def search_wolframalpha(query):
    try:
        res = wolfram_client.query(query)
        return next(res.results).text
    except:
        return "Could not compute answer."

def query_openai(prompt):
    openai.organization = os.getenv('OPENAI_ORG')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    try:
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0.3, max_tokens=80)
        return response.choices[0].text.strip()
    except:
        return "OpenAI request failed."

# --- Web and Navigation ---
def open_website(site):
    urls = {"youtube": "https://youtube.com", "google": "https://google.com", "facebook": "https://facebook.com"}
    url = urls.get(site.lower(), f"https://{site}.com")
    webbrowser.open(url)
    speak(f"Opening {site}")

def open_google_maps(location):
    url = f"https://www.google.com/maps/dir/?api=1&destination={location.replace(' ', '+')}"
    webbrowser.open(url)
    speak(f"Navigating to {location}")

# --- Time and Weather ---
def get_time():
    now = datetime.now().strftime("%H:%M:%S")
    speak(f"Current time: {now}")

def get_date():
    today = datetime.now().strftime("%Y-%m-%d")
    speak(f"Today's date: {today}")

def get_weather(city):
    api_key = "3H56DZ5P5FUNGF99MBJSKNBE2"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?key={api_key}&unitGroup=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if "days" in data:
            current = data["days"][0]
            speak(f"In {city}, {current['temp']}°C, {current['conditions']}. Humidity: {current['humidity']}%.")
        else:
            speak("Weather data not found.")
    except Exception as e:
        print("Error fetching weather:", e)
        speak("Weather fetch failed.")

# --- Command Handler ---
def handle_query(query):
    query = query.lower().strip()
    if not query:
        speak("Please say something.")
        return True

    if 'wikipedia' in query:
        result = search_wikipedia(query.replace('wikipedia', '').strip())
        speak(result)
    elif 'weather' in query:
        speak("Which city?")
        city = listen()
        get_weather(city)
    elif 'open' in query:
        site = query.replace('open', '').strip()
        open_website(site)
    elif 'navigate to' in query:
        location = query.replace('navigate to', '').strip()
        open_google_maps(location)
    elif 'time' in query:
        get_time()
    elif 'date' in query:
        get_date()
    elif 'ask' in query:
        speak("What's your question?")
        question = listen()
        answer = search_wolframalpha(question)
        speak(answer)
    elif 'openai' in query:
        speak("What's your prompt?")
        prompt = listen()
        response = query_openai(prompt)
        speak(response)
    elif 'play' in query and 'youtube' in query:
        song = query.replace('play', '').replace('on youtube', '').strip()
        pywhatkit.playonyt(song)
        speak(f"Playing {song} on YouTube")
    elif 'image' in query:
        speak("What image would you like?")
        prompt = listen()
        speak("AI or real image?")
        mode = listen()
        if any(k in mode for k in ['ai', 'artificial']):
            generate_ai_image(prompt)
        elif any(k in mode for k in ['real', 'internet']):
            search_real_image(prompt)
        else:
            speak("Invalid mode.")
    elif 'exit' in query or 'quit' in query:
        speak("Goodbye!")
        return False
    else:
        speak("Command not understood.")
    return True

# --- Main Loop ---
def main():
    speak("Hello, I'm your assistant. How can I help?")
    while True:
        query = listen()
        if not handle_query(query):
            break

if __name__ == "__main__":
    main()
