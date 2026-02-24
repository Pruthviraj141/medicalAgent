"""
answerer.py – MedBuddy conversational diagnostic pipeline.

Implements a multi-stage conversation flow:
  Stage 1: Greeting — friendly chat, no RAG needed
  Stage 2: Gathering — patient describes symptoms, ask follow-ups
  Stage 3: Diagnosis — enough info collected, provide assessment with severity

Dynamically detects conversation stage, manages evidence retrieval,
and ensures natural doctor-patient interaction before diagnosis.
"""
import json
import re
from app.retriever import hybrid_retrieve, detect_query_type
from app.memory import (
    get_short_memory,
    recall_long_memory,
    add_to_long_memory,
    add_to_short_memory,
)
from app.llm_client import llm_generate, llm_generate_async
from app.profiles import PROFILE
from app.cache import cache_get, cache_set


# ── emergency red-flag detection ──────────────────────────────────────

_EMERGENCY_PHRASES = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "severe breathing", "difficulty breathing", "shortness of breath",
    "vomiting blood", "coughing blood", "blood in vomit",
    "severe bleeding", "heavy bleeding", "won't stop bleeding",
    "loss of consciousness", "passed out", "unconscious", "fainted",
    "seizure", "convulsion", "stroke", "paralysis",
    "severe dehydration", "not urinating", "sunken eyes",
    "suicidal", "want to die", "kill myself",
    "overdose", "poisoning", "swallowed poison",
    "anaphylaxis", "throat swelling", "can't swallow",
]

_EMERGENCY_RESPONSE = {
    "answer": "⚠️ EMERGENCY — SEEK IMMEDIATE MEDICAL CARE. Please call emergency services or go to the nearest hospital right now.",
    "stage": "diagnosis",
    "possible_conditions": [],
    "recommended_action": "EMERGENCY",
    "severity": "Critical",
    "home_remedy": [],
    "followup_questions": [],
    "sources": [],
}


def _is_emergency(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _EMERGENCY_PHRASES)


# ── conversation stage detection ──────────────────────────────────────

_GREETING_PATTERNS = re.compile(
    r"^(h(i|ello|ey|owdy|ola)|good\s*(morning|afternoon|evening|night)"
    r"|what'?s?\s*up|yo+|namaste|hey\s*there|greetings"
    r"|how\s*are\s*you|how\s*do\s*you\s*do|sup|hii+|helo+)[\s!?.]*$",
    re.IGNORECASE,
)

_CASUAL_PATTERNS = re.compile(
    r"^(thanks?|thank\s*you|ok(ay)?|sure|yes|no|yep|nope|nah"
    r"|bye|goodbye|see\s*you|take\s*care|good\s*night"
    r"|who\s*are\s*you|what\s*can\s*you\s*do|help"
    r"|i'?m?\s*(fine|good|okay|great|alright|doing\s*well)"
    r"|nothing\s*much|not\s*much|i\s*don'?t\s*know|idk)[\s!?.]*$",
    re.IGNORECASE,
)

# Words that signal health concern
_HEALTH_KEYWORDS = [
    "pain", "ache", "hurt", "fever", "cough", "cold", "sore",
    "nausea", "vomit", "diarrhea", "rash", "itch", "swelling",
    "dizzy", "fatigue", "tired", "weak", "bleed", "infection",
    "burn", "headache", "stomach", "throat", "chest", "back",
    "joint", "muscle", "skin", "eye", "ear", "nose", "allergy",
    "sneez", "runny", "congest", "cramp", "bloat", "insomnia",
    "anxiety", "depress", "stress", "weight", "breath", "wheez",
    "symptom", "sick", "ill", "unwell", "disease", "condition",
    "diagnos", "medicat", "medicine", "tablet", "pill", "dose",
    "pregnant", "period", "blood", "pressure", "sugar", "diabetes",
    "asthma", "cancer", "heart", "liver", "kidney",
]


def _has_health_keywords(text: str) -> bool:
    """Check if text contains health-related keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _HEALTH_KEYWORDS)


def _detect_stage(question: str, short_ctx: list[dict]) -> str:
    """
    Detect conversation stage:
      "greeting"  — casual hi/hello, no health content
      "gathering" — health topic mentioned but need more info
      "diagnosis" — enough conversation turns with health info to diagnose
    """
    q_lower = question.strip().lower()
    turn_count = len(short_ctx)

    # Pure greeting with no history
    if turn_count == 0 and _GREETING_PATTERNS.match(question.strip()):
        return "greeting"

    # Casual chat responses
    if _CASUAL_PATTERNS.match(question.strip()):
        has_health_in_history = any(
            msg["role"] == "user" and _has_health_keywords(msg["text"])
            for msg in short_ctx
        )
        if not has_health_in_history:
            return "greeting"
        # They answered a follow-up — check if we have enough
        if turn_count >= 6:
            return "diagnosis"
        return "gathering"

    # Count user messages with health content
    health_turns = sum(
        1 for msg in short_ctx
        if msg["role"] == "user" and _has_health_keywords(msg["text"])
    )
    has_health_now = _has_health_keywords(question)
    if has_health_now:
        health_turns += 1

    # No health content at all
    if not has_health_now and health_turns == 0:
        return "greeting"

    # Decision: 3+ health data points OR 6+ total messages → diagnose
    if health_turns >= 3 or turn_count >= 6:
        return "diagnosis"

    # Health mentioned but not enough info yet
    if has_health_now or health_turns > 0:
        return "gathering"

    return "greeting"


def _get_accumulated_symptoms(short_ctx: list[dict], current_question: str) -> str:
    """Gather all health-related user messages for context."""
    symptoms = []
    for msg in short_ctx:
        if msg["role"] == "user" and _has_health_keywords(msg["text"]):
            symptoms.append(msg["text"])
    if _has_health_keywords(current_question):
        symptoms.append(current_question)
    return " | ".join(symptoms) if symptoms else current_question


# ── source extraction helpers ─────────────────────────────────────────

def _collect_sources(candidates: list[tuple]) -> dict:
    source_files: list[str] = []
    book_plants: list[str] = []
    merck_topics: list[str] = []
    seen_files: set[str] = set()
    seen_plants: set[str] = set()

    for doc_id, text, _score in candidates:
        # Book chunks have "book::" prefix
        if doc_id.startswith("book::"):
            m = re.match(r"Plant:\s*(.+?)\s*\|", text)
            if m:
                name = m.group(1).strip()
                if name not in seen_plants:
                    book_plants.append(name)
                    seen_plants.add(name)
            continue

        # Text file chunks have filename as source in the ID
        # IDs look like: "The_Merck_clean.txt_abcd1234_0"
        # or "medical_knowledge.txt_abcd1234_0"
        fname = doc_id.rsplit("_", 2)[0] if "_" in doc_id else doc_id
        # Also try splitting on "::" for legacy IDs
        if "::" in doc_id:
            fname = doc_id.split("::")[0]

        if fname.lower().endswith((".txt", ".md")):
            if fname not in seen_files:
                source_files.append(fname)
                seen_files.add(fname)

    return {"files": source_files, "plants": book_plants, "merck_topics": merck_topics}


def _build_source_list(sources_info: dict) -> list[str]:
    out: list[str] = []
    for fname in sources_info["files"][:5]:
        out.append(fname)
    for plant in sources_info["plants"][:3]:
        out.append(f"book.json ({plant})")
    for topic in sources_info["merck_topics"][:5]:
        out.append(f"Merck: {topic}")
    return out if out else ["medical_knowledge.txt"]


# ── evidence block builder ────────────────────────────────────────────

def _build_evidence_block(candidates: list[tuple], n: int) -> str:
    if not candidates:
        return "No evidence retrieved."

    blocks: list[str] = []
    for i, (doc_id, text, _score) in enumerate(candidates[:n], 1):
        if doc_id.startswith("book::"):
            label = "Ayurvedic Book (book.json)"
        elif "merck" in doc_id.lower() or "The_Merck" in doc_id:
            label = "Merck Manual"
        else:
            # Extract filename from ID
            fname = doc_id.rsplit("_", 2)[0] if "_" in doc_id else doc_id
            if "::" in doc_id:
                fname = doc_id.split("::")[0]
            label = fname if fname else "medical_knowledge.txt"

        snippet = text[:800] if len(text) > 800 else text
        blocks.append(f"[Evidence {i} — {label}]\n{snippet}")

    return "\n\n---\n\n".join(blocks)


# ── conversation context ─────────────────────────────────────────────

def _build_conversation_context(short_ctx: list[dict]) -> str:
    if not short_ctx:
        return ""
    lines = []
    for msg in short_ctx[-10:]:
        role = "Patient" if msg["role"] == "user" else "MedBuddy"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)


# ── stage-aware prompt builder ────────────────────────────────────────

def _build_prompt(
    question: str,
    conversation_history: str,
    evidence_block: str,
    long_texts: list[str],
    sources_info: dict,
    stage: str,
    query_type: str,
    accumulated_symptoms: str,
) -> str:
    parts: list[str] = []

    # conversation history
    if conversation_history:
        parts.append(f"=== Conversation so far ===\n{conversation_history}")

    # patient memory from past sessions
    if long_texts:
        parts.append(
            "=== Patient history (past visits) ===\n"
            + "\n".join(f"• {t}" for t in long_texts[:3])
        )

    # evidence (only for health-related stages)
    if stage in ("gathering", "diagnosis") and evidence_block:
        parts.append(f"=== Evidence from medical sources ===\n{evidence_block}")

    # source summary
    source_summary_parts = []
    if sources_info.get("merck_topics"):
        source_summary_parts.append(
            "Merck topics found: " + ", ".join(sources_info["merck_topics"][:6])
        )
    if sources_info.get("plants"):
        source_summary_parts.append(
            "Ayurvedic remedies available: " + ", ".join(sources_info["plants"][:4])
        )
    if source_summary_parts:
        parts.append("=== Available sources ===\n" + "\n".join(source_summary_parts))

    context = "\n\n".join(parts) if parts else "(No prior context)"

    # stage-specific instruction
    if stage == "greeting":
        instruction = (
            "STAGE: GREETING. The patient is saying hi or chatting casually. "
            "Respond warmly and friendly like a caring doctor. Ask how they're feeling or "
            "what brings them here today. "
            "Do NOT diagnose. Do NOT mention diseases. Just be friendly. "
            "Set stage='greeting' in JSON. possible_conditions=[], severity='', recommended_action=''."
        )
    elif stage == "gathering":
        instruction = (
            "STAGE: GATHERING. The patient has mentioned health concerns but you need more details.\n"
            f"Symptoms mentioned so far: {accumulated_symptoms}\n\n"
            "Acknowledge what they've shared empathetically. Then ask 2-3 follow-up questions. "
            "Choose from:\n"
            "  - How long have you had this? (duration)\n"
            "  - How bad is it — mild, moderate, or severe? (intensity)\n"
            "  - Any other symptoms alongside? (associated symptoms)\n"
            "  - Taking any medications currently? (medications)\n"
            "  - Any recent travel, injuries, or diet changes? (triggers)\n"
            "  - Does anything make it better or worse? (aggravating/relieving factors)\n"
            "Pick the MOST relevant questions you haven't already asked in the conversation. "
            "Do NOT diagnose yet. Set stage='gathering'. possible_conditions=[]."
        )
    else:  # diagnosis
        instruction = (
            "STAGE: DIAGNOSIS. You have gathered enough information.\n"
            f"All patient symptoms: {accumulated_symptoms}\n\n"
            "Provide your clinical assessment NOW:\n"
            "1. possible_conditions: 1-3 entries with confidence_percent and reason. NEVER empty.\n"
            "2. Link each condition to what the patient told you AND the evidence.\n"
            "3. severity: 'Low', 'Moderate', 'High', or 'Critical'.\n"
            "4. recommended_action: 'Home care' / 'See doctor within 48 hours' / 'See doctor today' / 'EMERGENCY'.\n"
            "5. Include home_remedy if Ayurvedic evidence exists.\n"
            "6. Start your answer referencing the conversation: 'Based on what you've described...'\n"
            "Set stage='diagnosis'. followup_questions can be empty or have 1 item."
        )

    return f"""Patient says: "{question}"

{context}

{instruction}

Output your response as friendly text followed by the mandatory ```json``` block. The JSON must be valid."""


# ── JSON response parser ─────────────────────────────────────────────

def _parse_response_json(raw_response: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    depth = 0
    start = -1
    for i, ch in enumerate(raw_response):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = raw_response[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and ("answer" in obj or "stage" in obj):
                        return obj
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


def _build_fallback_response(raw_text: str, source_list: list[str], stage: str) -> dict:
    clean = re.sub(r"```(?:json)?.*?```", "", raw_text, flags=re.DOTALL).strip()
    if not clean:
        clean = raw_text.strip()

    return {
        "answer": clean[:500],
        "stage": stage,
        "possible_conditions": [],
        "recommended_action": "" if stage != "diagnosis" else "See doctor if symptoms persist",
        "severity": "",
        "home_remedy": [],
        "followup_questions": [],
        "sources": source_list if stage == "diagnosis" else [],
    }


def _normalize_response(parsed: dict, source_list: list[str], stage: str) -> dict:
    detected_stage = parsed.get("stage", stage)

    result = {
        "answer": str(parsed.get("answer", "")),
        "stage": detected_stage,
        "possible_conditions": parsed.get("possible_conditions", [])[:3],
        "recommended_action": str(parsed.get("recommended_action", "")),
        "severity": str(parsed.get("severity", "")),
        "home_remedy": parsed.get("home_remedy", []),
        "followup_questions": parsed.get("followup_questions", [])[:3],
        "sources": parsed.get("sources", []),
    }

    # Enforce stage rules
    if detected_stage in ("greeting", "gathering"):
        result["possible_conditions"] = []
        result["severity"] = ""
    elif detected_stage == "diagnosis":
        if not result["sources"]:
            result["sources"] = source_list
        if not result["severity"]:
            result["severity"] = "Moderate"

    return result


def _extract_friendly_text(raw: str) -> str:
    text = re.sub(r"```(?:json)?.*?```", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"\{[^{}]*\"answer\".*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:400] if text else ""


# ── main async pipeline ──────────────────────────────────────────────

async def compose_answer_async(session_id: str, user_id: str, question: str):
    """
    Full async MedBuddy conversational pipeline:

    1. Emergency check → immediate escalation
    2. Load conversation history & detect stage
    3. Skip RAG for greetings; do hybrid retrieval for health queries
    4. Build stage-aware prompt
    5. LLM call → parse structured JSON
    6. Normalize & enforce stage rules
    7. Memory persistence
    """
    # 1) emergency — always first
    if _is_emergency(question):
        add_to_short_memory(session_id, "user", question, user_id=user_id)
        add_to_short_memory(session_id, "assistant",
                            _EMERGENCY_RESPONSE["answer"], user_id=user_id)
        return _EMERGENCY_RESPONSE, []

    # 2) conversation context & stage detection
    short_ctx = get_short_memory(session_id)
    stage = _detect_stage(question, short_ctx)
    conversation_history = _build_conversation_context(short_ctx)
    accumulated_symptoms = _get_accumulated_symptoms(short_ctx, question)

    # long-term memory
    long_mem = recall_long_memory(user_id, question, top_k=3)
    long_texts = [lm["text"] for lm in long_mem]

    final_docs = PROFILE["final_context_docs"]
    candidates = []
    sources_info = {"files": [], "plants": [], "merck_topics": []}
    source_list = []
    evidence_block = ""

    # 3) RAG retrieval only for health-related stages
    if stage in ("gathering", "diagnosis"):
        retrieval_query = accumulated_symptoms if stage == "diagnosis" else question
        candidates = hybrid_retrieve(retrieval_query)
        sources_info = _collect_sources(candidates[:final_docs * 2])
        source_list = _build_source_list(sources_info)
        evidence_block = _build_evidence_block(candidates, final_docs)

    query_type = detect_query_type(question) if stage != "greeting" else "general"

    # 4) build prompt
    prompt_text = _build_prompt(
        question, conversation_history, evidence_block,
        long_texts, sources_info, stage, query_type, accumulated_symptoms,
    )

    # 5) LLM call
    raw_response = await llm_generate_async(
        [{"role": "user", "content": prompt_text}],
        temperature=PROFILE["llm_temperature"],
        max_tokens=PROFILE["llm_max_tokens"],
    )

    # 6) parse and normalize
    parsed = _parse_response_json(raw_response)
    if parsed:
        structured = _normalize_response(parsed, source_list, stage)
    else:
        structured = _build_fallback_response(raw_response, source_list, stage)

    if not structured["answer"]:
        friendly = _extract_friendly_text(raw_response)
        structured["answer"] = friendly if friendly else raw_response[:400]

    # 7) memory persistence
    add_to_short_memory(session_id, "user", question, user_id=user_id)
    add_to_long_memory(user_id, "user", question)

    answer_for_memory = structured["answer"]
    add_to_short_memory(session_id, "assistant", answer_for_memory, user_id=user_id)
    add_to_long_memory(user_id, "assistant", answer_for_memory)

    return structured, candidates


# ── sync fallback ─────────────────────────────────────────────────────

def compose_answer(session_id: str, user_id: str, question: str):
    """Synchronous version of the conversational pipeline."""
    if _is_emergency(question):
        add_to_short_memory(session_id, "user", question, user_id=user_id)
        add_to_short_memory(session_id, "assistant",
                            _EMERGENCY_RESPONSE["answer"], user_id=user_id)
        return _EMERGENCY_RESPONSE, []

    short_ctx = get_short_memory(session_id)
    stage = _detect_stage(question, short_ctx)
    conversation_history = _build_conversation_context(short_ctx)
    accumulated_symptoms = _get_accumulated_symptoms(short_ctx, question)

    long_mem = recall_long_memory(user_id, question, top_k=3)
    long_texts = [lm["text"] for lm in long_mem]

    final_docs = PROFILE["final_context_docs"]
    candidates = []
    sources_info = {"files": [], "plants": [], "merck_topics": []}
    source_list = []
    evidence_block = ""

    if stage in ("gathering", "diagnosis"):
        retrieval_query = accumulated_symptoms if stage == "diagnosis" else question
        candidates = hybrid_retrieve(retrieval_query)
        sources_info = _collect_sources(candidates[:final_docs * 2])
        source_list = _build_source_list(sources_info)
        evidence_block = _build_evidence_block(candidates, final_docs)

    query_type = detect_query_type(question) if stage != "greeting" else "general"

    prompt_text = _build_prompt(
        question, conversation_history, evidence_block,
        long_texts, sources_info, stage, query_type, accumulated_symptoms,
    )

    raw_response = llm_generate(
        [{"role": "user", "content": prompt_text}],
        temperature=PROFILE["llm_temperature"],
        max_tokens=PROFILE["llm_max_tokens"],
    )

    parsed = _parse_response_json(raw_response)
    if parsed:
        structured = _normalize_response(parsed, source_list, stage)
    else:
        structured = _build_fallback_response(raw_response, source_list, stage)

    if not structured["answer"]:
        friendly = _extract_friendly_text(raw_response)
        structured["answer"] = friendly if friendly else raw_response[:400]

    add_to_short_memory(session_id, "user", question, user_id=user_id)
    add_to_long_memory(user_id, "user", question)

    answer_for_memory = structured["answer"]
    add_to_short_memory(session_id, "assistant", answer_for_memory, user_id=user_id)
    add_to_long_memory(user_id, "assistant", answer_for_memory)

    return structured, candidates
