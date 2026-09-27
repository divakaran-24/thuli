from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found")
    exit()

print("✅ API key found")

client = genai.Client(api_key=api_key)

print("\nAvailable Gemini models:\n")

for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(model.name)