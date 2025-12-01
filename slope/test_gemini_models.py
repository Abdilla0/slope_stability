import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print("Available Gemini Models:\n")
for model in genai.list_models():
    print("-", model.name)
