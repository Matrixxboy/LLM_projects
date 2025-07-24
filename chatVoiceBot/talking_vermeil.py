import torch
import sounddevice as sd
import requests
import textwrap
import speech_recognition as sr
from TTS.api import TTS

# 🧠 OLLAMA Config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "vermeil" # You can also use mistral, llama3, etc.

# 🎤 Load the TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# 🎙️ Setup speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()

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

        # 🔚 Exit if needed
        if user_prompt.lower() in ['exit', 'quit']:
            print("👋 Goodbye.")
            break

        # 🧠 Generate response from Ollama
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": user_prompt, "stream": False}
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
