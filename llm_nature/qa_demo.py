from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .word_ngram import WordNGramModel, tokenize

@dataclass
class QAPair:
    question: str
    answer: str

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

def best_answer_for_question(
    model: WordNGramModel,
    qa_pairs: List[QAPair],
    question: str,
    sim_threshold: float = 0.10,
) -> Tuple[str, float, float, float]:
    q_star_tokens = question_tokens(question)
    sims: List[float] = []
    for qa in qa_pairs:
        sims.append(jaccard(q_star_tokens, question_tokens(qa.question)))
    max_sim = max(sims) if sims else 0.0
    candidates: List[Tuple[QAPair, float]] = []
    for qa, s in zip(qa_pairs, sims):
        if s >= sim_threshold * max(1.0, max_sim):
            candidates.append((qa, s))
    if not candidates:
        candidates = list(zip(qa_pairs, sims))
    best_ans = ""
    best_score = float("-inf")
    best_sim = 0.0
    best_lm = float("-inf")
    for qa, s in candidates:
        prompt = f"Q: {question} A: {qa.answer}"
        lm_score = model.log_prob_text(prompt)
        if lm_score > best_score:
            best_score = lm_score
            best_ans = qa.answer
            best_sim = s
            best_lm = lm_score
    return best_ans, best_lm, best_sim, max_sim

def main() -> None:
    raw = load_corpus_text()
    qa_pairs = parse_qa_pairs(raw)
    print(f"Loaded {len(qa_pairs)} QA pairs from data/qa_corpus.txt")
    model = WordNGramModel.from_corpus(raw, k=4, alpha=0.5)
    questions = [
        "What is a large language model?",
        "What does conditional next-token generator mean?",
        "What is perplexity?",
        "What is cross-entropy in this setting?",
        "Why can a high-order n-gram look intelligent?",
        "Why does more data usually help these models?",
    ]
    for q in questions:
        print()
        print("Q:", q)
        ans, lm_score, sim, max_sim = best_answer_for_question(model, qa_pairs, q)
        print("MODEL (retrieval + LM re-ranking):")
        print("A:", ans)
        print(f"[log p ≈ {lm_score:.2f}, sim={sim:.2f}, max_sim={max_sim:.2f}]")

if __name__ == "__main__":
    main()
