"""LLM client utilities."""
import requests
from app.config import OPENROUTER_API_KEY, LLM_MODEL

class OpenRouterLLM:

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a medical AI assistant. Provide explainable answers."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.url, headers=headers, json=data)

        return response.json()["choices"][0]["message"]["content"]

llm = OpenRouterLLM()