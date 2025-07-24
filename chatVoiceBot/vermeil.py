import torch
import sounddevice as sd
import requests
import textwrap
from TTS.api import TTS

# 🧠 OLLAMA Config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"  # Or "mistral", "llama3", etc.

# 🎤 Load the TTS model (once)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

# 🔁 Conversation loop
while True:
    try:
        # 💬 Ask the user
        user_prompt = input("\n🧑 You: ")

        # 🔚 Exit if the user types quit/exit
        if user_prompt.lower() in ['exit', 'quit']:
            print("👋 Goodbye.")
            break

        # 🧠 Send prompt to Ollama
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": user_prompt, "stream": False}
        )
        reply = response.json().get("response", "").strip()

        # 💬 Print response
        print("\n🤖 Vermeil:\n", textwrap.fill(reply, width=80))

        # 🔊 Speak response
        audio = tts.tts(reply)
        sd.play(audio, samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()

    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")
        break
    except Exception as e:
        print("⚠️ Error:", e)
