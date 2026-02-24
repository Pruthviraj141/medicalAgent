import requests

API_KEY = ""  # <-- Put your API key here

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
# nvidia/nemotron-nano-9b-v2:free
data = {
    "model": "openai/gpt-3.5-turbo",  # You can change model if needed
    "messages": [
        {"role": "user", "content": "Hi"}
    ]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("✅ API Key is working!")
    print("Response:", response.json()["choices"][0]["message"]["content"])
else:
    print("❌ API Key failed!")
    print("Status Code:", response.status_code)
    print("Error:", response.text)
