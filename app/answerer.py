"""
answerer.py – compose the final friendly-doctor answer with confidence & follow-ups
"""
from app.retriever import hybrid_retrieve
from app.memory import (
    get_short_memory,
    recall_long_memory,
    add_to_long_memory,
    add_to_short_memory,
)
from app.llm_client import llm_generate


def compose_answer(session_id: str, user_id: str, question: str):
    """
    Full pipeline:
    1. Gather short-term conversation memory
    2. Hybrid retrieve (vector + BM25 + rerank)
    3. Recall relevant long-term memory
    4. Build a structured prompt → LLM
    5. Store conversation turn in both memory stores
    """

    # 1) short-term memory
    short_ctx = get_short_memory(session_id)
    short_text = "\n".join(f"{m['role']}: {m['text']}" for m in short_ctx)

    # 2) hybrid retrieval
    candidates = hybrid_retrieve(question, top_k=8)
    top_texts = [t[1] for t in candidates[:5]]

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

    context = "\n\n".join(context_blocks)

    prompt_user = f"""
User question:
{question}

Context:
{context}

Task:
1) Provide a friendly, empathetic diagnostic-style explanation (like a doctor friend). Use 1-2 emojis.
2) Give possible conditions (ranked), short reasoning bullets connecting symptoms to conditions.
3) Provide a clear confidence percentage (0-100%) and why (brief).
4) Provide 1-3 natural follow-up questions to refine diagnosis.
5) Provide a short recommended next action (home care vs see doctor vs emergency) and cite the source(s) by short filename or excerpt.

Respond in JSON with keys: answer, possible_conditions, confidence_percent, followup_questions, recommended_action, sources.
"""

    response = llm_generate(
        [{"role": "user", "content": prompt_user}],
        temperature=0.25,
        max_tokens=600,
    )

    # 5) store conversation
    add_to_short_memory(session_id, "user", question)
    add_to_long_memory(user_id, "user", question)
    add_to_short_memory(session_id, "assistant", response)
    add_to_long_memory(user_id, "assistant", response)

    return response, candidates
