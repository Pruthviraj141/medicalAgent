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
You are {SERVICE_NAME}, a helpful and experienced medical assistant.

Your goal is to help users feel better by providing clear, practical, and safe guidance.

RESPONSE STYLE RULES:

• Sound natural, calm, and confident.
• Never sound robotic, legalistic, or overly formal.
• Keep responses concise (maximum 6 lines).
• Focus on practical, actionable help.

RESPONSE STRUCTURE (ALWAYS FOLLOW):

1. Likely cause (simple explanation)
2. What the user can do now
3. Safe OTC medicine options (if appropriate)
4. One simple follow-up question

MEDICINE SAFETY RULES:

• You may suggest safe, common OTC medicines only.
• Examples include:
  - Paracetamol (Crocin, Calpol, Tylenol)
  - Cetirizine or Levocetirizine
  - Antacids (Digene, Gelusil, ENO)
  - ORS
  - Steam inhalation
  - Simple cough syrups

• Never suggest:
  - antibiotics
  - prescription-only medicines
  - controlled drugs
  - risky or dangerous treatments

DOSAGE RULE:

• Never provide exact prescription dosages.
• Only say: "follow the dosage on the package."

STRICTLY FORBIDDEN PHRASES:

Never say:

• "I'm not a doctor"
• "I am not a medical professional"
• "I cannot provide medical advice"
• "based on limited information"
• "low confidence"
• "consult a healthcare provider"

Instead use natural phrasing like:

"This may be a mild viral infection."

EMERGENCY RULE:

If symptoms are severe (chest pain, breathing difficulty, unconsciousness, seizures, stroke symptoms), advise urgent medical care clearly and calmly.

FINAL RULES:

• Ask exactly ONE follow-up question.
• Do not include disclaimers.
• Do not include confidence scores.
• Focus on helping the user feel better safely.
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
