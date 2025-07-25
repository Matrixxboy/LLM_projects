import torch
import sounddevice as sd
import requests
import textwrap
import speech_recognition as sr
from TTS.api import TTS
import geocoder
import datetime

# 🧠 OLLAMA Config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "vermeil"

# 🎤 Load the TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# 🎙️ Setup speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

# 🌤️ Weather fetch function
def get_weather():
    try:
        # Get location based on IP
        g = geocoder.ip('me')
        lat, lon = g.latlng
        # Use Open-Meteo (no API key required)
        weather_api = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(weather_api).json()
        temp = res["current_weather"]["temperature"]
        wind = res["current_weather"]["windspeed"]
        return f"Current weather: {temp}°C, Wind speed: {wind} km/h."
    except Exception as e:
        return "Weather data not available."

# 🔁 Conversation loop
while True:
    try:
        print("\n🎤 Speak now (or say 'quit' to exit)...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_input = recognizer.listen(source)

        # 🔊 Convert speech to text
        user_prompt = recognizer.recognize_google(audio_input)
        print(f"\n🧑 You said: {user_prompt}")

        if user_prompt.lower() in ['exit', 'quit']:
            print("👋 Goodbye.")
            break

        # 📡 Get weather & other context
        weather_info = get_weather()
        now = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

        # 🧠 Final prompt with context injection
        context_prompt = (
            f"[System Info]\nTime: {now}\n{weather_info}\n\n"
            f"[User]: {user_prompt}\n[Assistant]:"
        )

        # 🧠 Generate response from Ollama
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": context_prompt, "stream": False}
        )
        reply = response.json().get("response", "").strip()

        # 💬 Print and speak
        print("\n🤖 Vermeil:\n", textwrap.fill(reply, width=80))
        audio = tts.tts(reply)
        sd.play(audio, samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()

    except sr.UnknownValueError:
        print("⚠️ Could not understand your voice. Please try again.")
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")
        break
    except Exception as e:
        print("⚠️ Error:", e)
