"""
answerer.py – compose the final friendly-doctor answer with natural conversation,
               follow-ups, source attribution, and memory awareness.
"""
import re
from app.retriever import hybrid_retrieve
from app.memory import (
    get_short_memory,
    recall_long_memory,
    add_to_long_memory,
    add_to_short_memory,
)
from app.llm_client import llm_generate, llm_generate_async
from app.profiles import PROFILE
from app.cache import cache_get, cache_set


def _extract_book_sources(candidates: list[tuple]) -> list[str]:
    """
    Pull plant names from book.json candidate IDs for source attribution.
    """
    sources: list[str] = []
    seen: set[str] = set()
    for doc_id, text, _score in candidates:
        if not doc_id.startswith("book::"):
            continue
        m = re.match(r"Plant:\s*(.+?)\s*\|", text)
        if m:
            name = m.group(1).strip()
            if name not in seen:
                sources.append(name)
                seen.add(name)
    return sources


def _build_conversation_context(short_ctx: list[dict]) -> str:
    """
    Format conversation history as a natural chat log so the LLM
    understands the flow and can reference prior details.
    """
    if not short_ctx:
        return ""

    lines = []
    for msg in short_ctx[-8:]:  # last 8 messages max for context window
        role = "Patient" if msg["role"] == "user" else "You"
        lines.append(f"{role}: {msg['text']}")

    return "\n".join(lines)


def _build_prompt(question: str, conversation_history: str,
                  top_texts: list[str], long_texts: list[str],
                  book_sources: list[str], is_first_message: bool) -> str:
    """
    Build a natural, context-aware prompt that produces human-like responses.
    """
    parts: list[str] = []

    # conversation history (most important for continuity)
    if conversation_history:
        parts.append(
            f"Previous conversation:\n{conversation_history}"
        )

    # long-term memory (past sessions)
    if long_texts:
        parts.append(
            "What you remember about this patient from past visits:\n"
            + "\n".join(f"- {t}" for t in long_texts[:3])
        )

    # retrieved medical evidence
    if top_texts:
        parts.append(
            "Medical knowledge to base your answer on:\n"
            + "\n---\n".join(top_texts)
        )

    # Ayurvedic sources
    if book_sources:
        parts.append(
            "Ayurvedic remedies available: " + ", ".join(book_sources)
            + "\n(Mention these naturally if relevant — include the plant name and how to use it)"
        )

    context = "\n\n".join(parts)

    # different instruction based on conversation stage
    if is_first_message:
        instruction = (
            "This is the patient's FIRST message. Respond warmly, give an initial assessment, "
            "and ask 1-2 caring follow-up questions to understand their situation better "
            "(like 'How long have you been feeling this way?' or 'Is the pain sharp or dull?')."
        )
    else:
        instruction = (
            "Continue the conversation naturally. Reference what the patient told you before. "
            "Build on previous information. Give a more specific assessment now that you know more. "
            "If you have enough info, give clear advice. Ask ONE more follow-up only if needed."
        )

    return f"""Patient says: "{question}"

{context}

{instruction}

Respond as a caring doctor friend — natural, warm, 4-6 sentences. No bullet lists, no headers, no confidence scores."""


async def compose_answer_async(session_id: str, user_id: str, question: str):
    """
    Full async pipeline:
    1. Check conversation history (is this first message?)
    2. Hybrid retrieve (vector + BM25 + rerank)
    3. Recall relevant long-term memory
    4. Build natural conversation prompt → LLM (async)
    5. Store conversation turn in both memory stores
    """
    final_docs = PROFILE["final_context_docs"]

    # 1) conversation history
    short_ctx = get_short_memory(session_id)
    conversation_history = _build_conversation_context(short_ctx)
    is_first_message = len(short_ctx) == 0

    # 2) hybrid retrieval
    candidates = hybrid_retrieve(question)
    top_texts = [t[1] for t in candidates[:final_docs]]

    # 3) long-term memory (past sessions with this user)
    long_mem = recall_long_memory(user_id, question, top_k=3)
    long_texts = [lm["text"] for lm in long_mem]

    # 4) build natural prompt
    book_sources = _extract_book_sources(candidates[:final_docs])
    prompt_text = _build_prompt(
        question, conversation_history, top_texts,
        long_texts, book_sources, is_first_message
    )

    response = await llm_generate_async(
        [{"role": "user", "content": prompt_text}],
        temperature=PROFILE["llm_temperature"],
        max_tokens=PROFILE["llm_max_tokens"],
    )

    # 5) store conversation in memory
    add_to_short_memory(session_id, "user", question, user_id=user_id)
    add_to_long_memory(user_id, "user", question)
    add_to_short_memory(session_id, "assistant", response, user_id=user_id)
    add_to_long_memory(user_id, "assistant", response)

    return response, candidates


def compose_answer(session_id: str, user_id: str, question: str):
    """
    Synchronous version of the full pipeline (fallback for non-async callers).
    """
    final_docs = PROFILE["final_context_docs"]

    # 1) conversation history
    short_ctx = get_short_memory(session_id)
    conversation_history = _build_conversation_context(short_ctx)
    is_first_message = len(short_ctx) == 0

    # 2) hybrid retrieval
    candidates = hybrid_retrieve(question)
    top_texts = [t[1] for t in candidates[:final_docs]]

    # 3) long-term memory
    long_mem = recall_long_memory(user_id, question, top_k=3)
    long_texts = [lm["text"] for lm in long_mem]

    # 4) build natural prompt
    book_sources = _extract_book_sources(candidates[:final_docs])
    prompt_text = _build_prompt(
        question, conversation_history, top_texts,
        long_texts, book_sources, is_first_message
    )

    response = llm_generate(
        [{"role": "user", "content": prompt_text}],
        temperature=PROFILE["llm_temperature"],
        max_tokens=PROFILE["llm_max_tokens"],
    )

    # 5) store conversation
    add_to_short_memory(session_id, "user", question, user_id=user_id)
    add_to_long_memory(user_id, "user", question)
    add_to_short_memory(session_id, "assistant", response, user_id=user_id)
    add_to_long_memory(user_id, "assistant", response)

    return response, candidates
