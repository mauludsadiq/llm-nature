from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .word_ngram import WordNGramModel, tokenize

@dataclass
class QAPair:
    question: str
    answer: str

FOIL_ANSWER = (
    "A large language model is a conscious agent that understands meaning and "
    "reasons about the world like a human."
)

CORRECT_ANS_FOR_Q = {
    "What is a large language model?": (
        "A large language model is a conditional next-token generator trained "
        "to minimize cross-entropy over a huge text corpus."
    ),
    "What does conditional next-token generator mean?": (
        "It means the model maps a context sequence of tokens to a probability "
        "distribution over the next token."
    ),
    "What is perplexity?": (
        "Perplexity is the exponential of the cross-entropy and measures the "
        "effective branching factor of the model."
    ),
    "What is cross-entropy in this setting?": (
        "Cross-entropy is the average negative log-likelihood of the true next "
        "token under the model's predictive distribution."
    ),
    "Why can a high-order n-gram look intelligent?": (
        "Because when trained on enough structured text, it can reproduce "
        "locally coherent patterns that resemble understanding."
    ),
    "Why does more data usually help these models?": (
        "More data reduces variance in the estimated conditional distributions "
        "and lets the model support higher effective Markov order."
    ),
}

def load_corpus_text() -> str:
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "qa_corpus.txt"
    return path.read_text(encoding="utf-8")

def parse_qa_pairs(text: str) -> List[QAPair]:
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    pairs: List[QAPair] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        q_line = None
        a_line = None
        for ln in lines:
            if ln.startswith("Q:"):
                q_line = ln[len("Q:") :].strip()
            elif ln.startswith("A:"):
                a_line = ln[len("A:") :].strip()
        if q_line and a_line:
            pairs.append(QAPair(question=q_line, answer=a_line))
    return pairs

def question_tokens(q: str) -> set:
    toks = tokenize(q)
    return {t.lower() for t in toks if any(c.isalpha() for c in t)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def inspect_question(
    model: WordNGramModel,
    qa_pairs: List[QAPair],
    question: str,
) -> Tuple[bool, bool]:
    q_star = question_tokens(question)
    rows: List[Tuple[float, float, str, str]] = []
    for qa in qa_pairs:
        sim = jaccard(q_star, question_tokens(qa.question))
        prompt = f"Q: {question} A: {qa.answer}"
        lm_score = model.log_prob_text(prompt)
        total = lm_score
        rows.append((total, sim, qa.question, qa.answer))
    rows.sort(key=lambda r: r[0], reverse=True)

    print(f"\n=== QUESTION: {question!r} ===")
    for rank, (total, sim, q_train, ans) in enumerate(rows, start=1):
        tag = ""
        if ans == FOIL_ANSWER:
            tag = " [FOIL]"
        elif CORRECT_ANS_FOR_Q.get(question) == ans:
            tag = " [CORRECT]"
        print(f"#{rank}: total={total:.2f}, sim={sim:.2f}{tag}")
        print(f"  train Q: {q_train}")
        print(f"  train A: {ans}")
        print()

    top_total, top_sim, top_q, top_a = rows[0]
    is_foil = (top_a == FOIL_ANSWER)
    is_correct = (CORRECT_ANS_FOR_Q.get(question) == top_a)
    return is_foil, is_correct

def main() -> None:
    raw = load_corpus_text()
    qa_pairs = parse_qa_pairs(raw)
    model = WordNGramModel.from_corpus(raw, k=4, alpha=0.5)

    questions = [
        "What is a large language model?",
        "What does conditional next-token generator mean?",
        "What is perplexity?",
        "What is cross-entropy in this setting?",
        "Why can a high-order n-gram look intelligent?",
        "Why does more data usually help these models?",
    ]

    summary: List[Tuple[str, bool, bool]] = []
    for q in questions:
        is_foil, is_correct = inspect_question(model, qa_pairs, q)
        summary.append((q, is_foil, is_correct))

    print("\n=== SUMMARY: top-1 classification ===")
    print("| Question | top-1 = FOIL | top-1 = CORRECT |")
    print("|----------|--------------:|---------------:|")
    for q, is_foil, is_correct in summary:
        print(f"| {q} | {int(is_foil)} | {int(is_correct)} |")

if __name__ == "__main__":
    main()
