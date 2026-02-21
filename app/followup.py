from app.llm import llm

def generate_followups(question, context):

    prompt = f"""
Based on medical question and context:

Question:
{question}

Context:
{context}

Generate 3 important follow-up diagnostic questions.
"""

    response = llm.generate(prompt)

    return response