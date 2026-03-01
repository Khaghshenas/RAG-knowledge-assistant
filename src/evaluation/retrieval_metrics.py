import re

def normalize(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text.strip()


def answer_in_text(answer, text):
    return normalize(answer) in normalize(text)


def recall_at_k(questions, retriever, reranker, k=5):
    hits = 0

    for q in questions:
        retrieved = retriever.search(q["question"], 20)[:20]
        reranked_chunks = reranker.rerank(q["question"], retrieved)
        final_chunks = reranked_chunks[:k]

        found = False
        for chunk in final_chunks:
            for ans in q["answers"]:
                if answer_in_text(ans, chunk["text"]):
                    found = True
                    break
            if found:
                break

        hits += int(found)

    return hits / len(questions)
