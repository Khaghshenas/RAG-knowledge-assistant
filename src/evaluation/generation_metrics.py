import re
from collections import Counter

def normalize(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text.strip()


def exact_match(pred, answers):
    pred = normalize(pred)
    return any(pred == normalize(a) for a in answers)


def f1_score(pred, answers):
    pred_tokens = normalize(pred).split()

    best_f1 = 0.0
    for ans in answers:
        ans_tokens = normalize(ans).split()

        common = Counter(pred_tokens) & Counter(ans_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ans_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1
