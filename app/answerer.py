"""
answerer.py – compose the final friendly-doctor answer with confidence,
               follow-ups, source attribution, and verification.
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

    Book chunk IDs look like: ``book::plant_3::treat_2``
    The chunk text starts with "Plant: <name> | Botanical name: …"
    """
    sources: list[str] = []
    seen: set[str] = set()
    for doc_id, text, _score in candidates:
        if not doc_id.startswith("book::"):
            continue
        # extract plant name from the text header
        m = re.match(r"Plant:\s*(.+?)\s*\|", text)
        if m:
            name = m.group(1).strip()
            if name not in seen:
                sources.append(name)
                seen.add(name)
    return sources


def _verify_answer(answer: str, sources: list[str]) -> str:
    """
    Quick heuristic: if the answer doesn't reference any source content,
    prepend a low-confidence warning.
    """
    source_text = " ".join(sources).lower()
    answer_lower = answer.lower()

    # check if at least one meaningful keyword from sources appears in answer
    keywords = [w for w in source_text.split() if len(w) > 5][:20]
    overlap = sum(1 for kw in keywords if kw in answer_lower)

    if keywords and overlap < 2:
        return (
            "⚠️ **LOW CONFIDENCE** — My answer may not be fully supported "
            "by the medical sources I found. Please verify with a doctor.\n\n"
            + answer
        )
    return answer


async def compose_answer_async(session_id: str, user_id: str, question: str):
    """
    Full async pipeline:
    1. Gather short-term conversation memory
    2. Hybrid retrieve (vector + BM25 + rerank)
    3. Recall relevant long-term memory
    4. Build a structured prompt → LLM (async)
    5. Verify answer against sources
    6. Store conversation turn in both memory stores
    """
    final_docs = PROFILE["final_context_docs"]

    # 1) short-term memory
    short_ctx = get_short_memory(session_id)
    short_text = "\n".join(f"{m['role']}: {m['text']}" for m in short_ctx)

    # 2) hybrid retrieval
    candidates = hybrid_retrieve(question)
    top_texts = [t[1] for t in candidates[:final_docs]]

    # 3) long-term memory
    long_mem = recall_long_memory(user_id, question, top_k=3)
    long_texts = [lm["text"] for lm in long_mem]

    # 4) build context blocks
    context_blocks: list[str] = []
    if short_text:
        context_blocks.append("Conversation so far:\n" + short_text)
    if long_texts:
        context_blocks.append("Relevant patient memory:\n" + "\n".join(long_texts))
    if top_texts:
        context_blocks.append(
            "Retrieved medical evidence (short excerpts):\n" + "\n---\n".join(top_texts)
        )

    # extract Ayurvedic book sources for attribution
    book_sources = _extract_book_sources(candidates[:final_docs])
    if book_sources:
        context_blocks.append(
            "Ayurvedic plant sources referenced: " + ", ".join(book_sources)
        )

    context = "\n\n".join(context_blocks)

    prompt_user = f"""
Question: {question}

Medical context:
{context}

Answer concisely (5-6 lines max):
- If Ayurvedic/herbal treatments are found, include the plant name, ailment, and specific treatment details
- Possible condition(s) with brief reasoning
- Confidence % and recommended action
- One follow-up question if needed
- Always mention the plant source when citing Ayurvedic remedies
"""

    response = await llm_generate_async(
        [{"role": "user", "content": prompt_user}],
        temperature=PROFILE["llm_temperature"],
        max_tokens=PROFILE["llm_max_tokens"],
    )

    # 5) verify against sources
    response = _verify_answer(response, top_texts)

    # 6) store conversation
    add_to_short_memory(session_id, "user", question)
    add_to_long_memory(user_id, "user", question)
    add_to_short_memory(session_id, "assistant", response)
    add_to_long_memory(user_id, "assistant", response)

    return response, candidates


def compose_answer(session_id: str, user_id: str, question: str):
    """
    Synchronous version of the full pipeline (fallback for non-async callers).
    """
    final_docs = PROFILE["final_context_docs"]

    # 1) short-term memory
    short_ctx = get_short_memory(session_id)
    short_text = "\n".join(f"{m['role']}: {m['text']}" for m in short_ctx)

    # 2) hybrid retrieval
    candidates = hybrid_retrieve(question)
    top_texts = [t[1] for t in candidates[:final_docs]]

    # 3) long-term memory
    long_mem = recall_long_memory(user_id, question, top_k=3)
    long_texts = [lm["text"] for lm in long_mem]

    # 4) build context blocks
    context_blocks: list[str] = []
    if short_text:
        context_blocks.append("Conversation so far:\n" + short_text)
    if long_texts:
        context_blocks.append("Relevant patient memory:\n" + "\n".join(long_texts))
    if top_texts:
        context_blocks.append(
            "Retrieved medical evidence (short excerpts):\n" + "\n---\n".join(top_texts)
        )

    # extract Ayurvedic book sources for attribution
    book_sources = _extract_book_sources(candidates[:final_docs])
    if book_sources:
        context_blocks.append(
            "Ayurvedic plant sources referenced: " + ", ".join(book_sources)
        )

    context = "\n\n".join(context_blocks)

    prompt_user = f"""
Question: {question}

Medical context:
{context}

Answer concisely (5-6 lines max):
- If Ayurvedic/herbal treatments are found, include the plant name, ailment, and specific treatment details
- Possible condition(s) with brief reasoning
- Confidence % and recommended action
- One follow-up question if needed
- Always mention the plant source when citing Ayurvedic remedies
"""

    response = llm_generate(
        [{"role": "user", "content": prompt_user}],
        temperature=PROFILE["llm_temperature"],
        max_tokens=PROFILE["llm_max_tokens"],
    )

    # 5) verify against sources
    response = _verify_answer(response, top_texts)

    # 6) store conversation
    add_to_short_memory(session_id, "user", question)
    add_to_long_memory(user_id, "user", question)
    add_to_short_memory(session_id, "assistant", response)
    add_to_long_memory(user_id, "assistant", response)

    return response, candidates
