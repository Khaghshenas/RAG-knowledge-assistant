from src.evaluation.generation_metrics import exact_match, f1_score

def evaluate_rag(rag_pipeline, eval_set, max_samples=None):
    ems, f1s = [], []

    for i, q in enumerate(eval_set):
        if max_samples and i >= max_samples:
            break

        result = rag_pipeline.answer(q["question"])
        pred = result["answer"]

        ems.append(exact_match(pred, q["answers"]))
        f1s.append(f1_score(pred, q["answers"]))

    return {
        "EM": sum(ems) / len(ems),
        "F1": sum(f1s) / len(f1s),
    }
