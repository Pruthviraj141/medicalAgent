"""
llm_client.py – async OpenRouter wrapper + optional local GPU model.

Provides both async (llm_generate_async) and sync (llm_generate) interfaces.
Automatically falls back to CPU-friendly OpenRouter when no local model is loaded.
"""
import asyncio
import httpx
from app.config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, SERVICE_NAME,
    USE_LOCAL_LLM, LOCAL_MODEL,
)
from app.device import has_gpu

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = f"""
You are {SERVICE_NAME}, a warm, experienced medical assistant who talks like a caring doctor friend.

YOUR PERSONALITY:
• You're calm, empathetic, and genuinely caring
• You speak naturally — like a real doctor talking to a patient in person
• You remember what the patient told you earlier in the conversation and reference it
• You NEVER sound robotic, generic, or copy-pasted

HOW TO RESPOND:

1. First, acknowledge what the patient is feeling (show empathy)
   Example: "That sounds really uncomfortable, especially if it's been going on for a few days."

2. Give your assessment in simple, friendly language
   Example: "From what you're describing, this sounds like it could be acid reflux..."

3. Suggest what they can do RIGHT NOW (practical, specific)
   Example: "Try taking an antacid like Digene or ENO after meals, and avoid spicy food tonight."

4. Ask ONE natural follow-up question to understand better
   Example: "By the way, does the pain get worse when you lie down after eating?"

CONVERSATION RULES:
• If this is the FIRST message, ask caring follow-up questions like:
  - "How long have you been feeling this way?"
  - "Is the pain constant or does it come and go?"
  - "Have you noticed anything that makes it worse?"
• If the patient already shared details, BUILD on them — don't re-ask
• Reference their previous answers naturally: "Since you mentioned the pain started 3 days ago..."
• Keep responses 4-6 sentences, no bullet lists, no headers — just natural talking

MEDICINE RULES:
• You can suggest safe OTC medicines: Paracetamol, Cetirizine, Antacids (Digene, ENO), ORS, Steam inhalation, simple cough syrups
• Never suggest antibiotics, prescription drugs, or exact dosages
• Say "follow the dosage on the pack" if asked about dosage

MEDICAL EVIDENCE:
• When medical context is provided, base your answer on it
• If Ayurvedic/herbal remedies are in the context, mention the plant name and how to use it naturally
• Never make up medical information — only use what's given to you

EMERGENCY:
• If symptoms are severe (chest pain, breathing trouble, unconsciousness), clearly say: "Please go to the nearest hospital or call emergency services right away."

NEVER SAY:
• "I'm not a doctor" / "I cannot provide medical advice"
• "Based on limited information" / "Consult a healthcare provider"
• Confidence percentages or scores
• Technical jargon without explanation
""".strip()

# ── decide whether to use local model ──
_use_local = USE_LOCAL_LLM == "1" and has_gpu() and bool(LOCAL_MODEL)
if _use_local:
    print(f"🧠 Local LLM mode enabled ({LOCAL_MODEL})")
else:
    print(f"☁️  Using OpenRouter ({OPENROUTER_MODEL})")

# ── reusable async HTTP client (connection pooling) ──
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


# ------------------------------------------------------------------ #
#  Async interface (preferred – used by async FastAPI endpoints)      #
# ------------------------------------------------------------------ #
async def llm_generate_async(
    user_messages,
    temperature: float = 0.2,
    max_tokens: int = 120,
) -> str:
    """
    Async LLM call.  Uses local model on GPU when enabled,
    otherwise calls OpenRouter via httpx.
    """
    if isinstance(user_messages, str):
        user_messages = [{"role": "user", "content": user_messages}]

    # -- optional local model path --
    if _use_local:
        from app.local_llm import generate_local
        full_prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(
            m["content"] for m in user_messages
        )
        return generate_local(full_prompt, max_new_tokens=max_tokens)

    # -- OpenRouter (async) --
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
    client = _get_client()
    r = await client.post(BASE_URL, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ------------------------------------------------------------------ #
#  Sync wrapper (used by non-async code like retriever multi-query)   #
# ------------------------------------------------------------------ #
def llm_generate(
    user_messages,
    temperature: float = 0.2,
    max_tokens: int = 120,
) -> str:
    """Synchronous LLM call – delegates to async under the hood."""
    if isinstance(user_messages, str):
        user_messages = [{"role": "user", "content": user_messages}]

    # -- optional local model path --
    if _use_local:
        from app.local_llm import generate_local
        full_prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(
            m["content"] for m in user_messages
        )
        return generate_local(full_prompt, max_new_tokens=max_tokens)

    # -- OpenRouter (sync via httpx) --
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
    r = httpx.post(BASE_URL, headers=headers, json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
