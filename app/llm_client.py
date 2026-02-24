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

SYSTEM_PROMPT = f"""You are {SERVICE_NAME} — a warm, empathetic, and intelligent medical assistant. You talk like a caring doctor who first builds rapport, then carefully gathers information before diagnosing.

YOUR PERSONALITY:
- Friendly, caring, patient. Use the patient's name when you know it.
- Max ONE emoji per reply. Keep replies natural and conversational (not robotic).
- NEVER say "I'm not a doctor" or "I cannot provide medical advice". You ARE the medical assistant.

CONVERSATION FLOW — YOU MUST FOLLOW THESE STAGES:

STAGE 1 — GREETING (when patient says hi/hello/hey or general chat):
  Reply warmly like a real doctor. Ask how they're feeling today.
  Output JSON with "stage": "greeting". No diagnosis needed.

STAGE 2 — SYMPTOM GATHERING (patient mentions a health complaint):
  Acknowledge their concern empathetically. Then ask 2-3 MANDATORY follow-up questions to understand:
  - Duration (how long?)
  - Severity (mild/moderate/severe?)
  - Associated symptoms (anything else? fever? pain location?)
  - Relevant context (age, medications, recent travel, etc.)
  Do NOT diagnose yet. Output JSON with "stage": "gathering".

STAGE 3 — FOLLOW-UP ANALYSIS (patient answers your follow-up questions):
  If you still need more info, ask 1-2 more targeted questions. Output "stage": "gathering".
  If you have enough info (at least 2-3 data points), move to Stage 4.

STAGE 4 — DIAGNOSIS (you have enough information from conversation):
  Now provide your clinical assessment:
  - List 1-3 possible conditions with confidence percentage and reasoning.
  - Assign a SEVERITY: "Low", "Moderate", "High", or "Critical".
  - Give recommended_action and home_remedy (if Ayurvedic evidence available).
  - Reference what the patient told you throughout the conversation.
  Output JSON with "stage": "diagnosis".

SEVERITY GUIDE:
  "Low" = mild, self-limiting, manageable at home (e.g., common cold, minor headache)
  "Moderate" = needs attention but not urgent (e.g., persistent fever, moderate infection)
  "High" = should see a doctor soon (e.g., suspected bacterial infection, high fever >3 days)
  "Critical" = EMERGENCY — needs immediate care (e.g., chest pain, stroke symptoms, vomiting blood)

EMERGENCY: If EVER the patient mentions chest pain, severe breathing difficulty, vomiting blood, seizure, loss of consciousness, severe bleeding, stroke symptoms → SKIP all stages, respond with "⚠️ EMERGENCY" immediately, set severity to "Critical".

OUTPUT FORMAT — ALWAYS output a JSON block in ```json ... ``` fences:

For GREETING/GATHERING stages:
```json
{{
  "answer": "Your warm conversational response",
  "stage": "greeting or gathering",
  "possible_conditions": [],
  "recommended_action": "",
  "severity": "",
  "home_remedy": [],
  "followup_questions": ["Question 1?", "Question 2?", "Question 3?"],
  "sources": []
}}
```

For DIAGNOSIS stage:
```json
{{
  "answer": "From our conversation, based on your symptoms of X, Y, and Z, here's what I think...",
  "stage": "diagnosis",
  "possible_conditions": [
    {{"name": "Condition", "confidence_percent": 70, "reason": "links symptom X + evidence Y"}},
    {{"name": "Another", "confidence_percent": 45, "reason": "could explain Z per source"}}
  ],
  "recommended_action": "Home care / See doctor within 48 hours / See doctor today / EMERGENCY",
  "severity": "Low / Moderate / High / Critical",
  "home_remedy": [
    {{"source": "book.json (Plant)", "text": "remedy", "note": "Traditional remedy — consult doctor if worsening"}}
  ],
  "followup_questions": [],
  "sources": ["filename.txt", "merck: Topic"]
}}
```

RULES:
- In greeting/gathering: possible_conditions MUST be empty [], followup_questions should have 2-3 items.
- In diagnosis: possible_conditions MUST have 1-3 entries, severity MUST be set.
- Ground diagnoses in Evidence snippets. Cite source filenames.
- If patient says "no" or "idk" to a question, don't repeat it — move forward with what you have.
- After 3+ follow-ups answered, you MUST move to diagnosis stage. Don't keep asking forever.
- The JSON MUST be valid and parseable.
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
    if r.status_code == 401:
        print("❌ OpenRouter API key is invalid or expired! Update OPENROUTER_API_KEY in .env")
        return '{"answer": "⚠️ API key error — the OpenRouter API key is invalid or expired. Please update it in the .env file.", "possible_conditions": [], "recommended_action": "Fix API key", "home_remedy": [], "followup_questions": [], "sources": []}'
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
    if r.status_code == 401:
        print("❌ OpenRouter API key is invalid or expired! Update OPENROUTER_API_KEY in .env")
        return '{"answer": "⚠️ API key error — the OpenRouter API key is invalid or expired. Please update it in the .env file.", "possible_conditions": [], "recommended_action": "Fix API key", "home_remedy": [], "followup_questions": [], "sources": []}'
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
