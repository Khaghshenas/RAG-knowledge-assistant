def build_prompt(question: str, retrieved_chunks: list) -> str:

    context = "\n\n".join(
        [f"- {chunk['text']}" for chunk in retrieved_chunks]
    )

    prompt = f"""
You are a question answering system.
Answer the question using ONLY the context below.
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt
