"""
llm_client.py – OpenRouter chat-completion wrapper with friendly-doctor system prompt
"""
import requests
from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, SERVICE_NAME

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = f"""
You are {SERVICE_NAME}, a friendly, empathetic medical assistant who speaks like a supportive doctor/friend.
- Keep responses conversational and warm, but medically responsible.
- Use short emojis to add friendliness (one or two max), e.g., 😊, 🤒, 👍, ⚠️.
- Always state reasoning briefly and cite sources (source: short excerpt or filename).
- When uncertain or high-risk symptoms appear, give clear escalation instructions.
- Ask 1–3 natural follow-up questions when necessary.
""".strip()


def llm_generate(user_messages, temperature=0.2, max_tokens=400):
    """
    Call the OpenRouter chat API.

    Parameters
    ----------
    user_messages : str | list[dict]
        A plain string (turned into a single user message) **or**
        a list of ``{"role": ..., "content": ...}`` dicts.
    temperature : float
    max_tokens : int

    Returns
    -------
    str – the assistant's reply text.
    """
    if isinstance(user_messages, str):
        user_messages = [{"role": "user", "content": user_messages}]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
