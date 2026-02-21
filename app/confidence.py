from app.llm import llm

def generate_confidence(question, context):

    prompt = f"""
Based on context and medical reasoning:

Question:
{question}

Context:
{context}

Give diagnosis confidence percentage only.
"""

    return llm.generate(prompt)
