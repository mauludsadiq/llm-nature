from __future__ import annotations

from pathlib import Path

from .word_ngram import WordNGramModel

def load_corpus() -> str:
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "chat_corpus.txt"
    return path.read_text(encoding="utf-8")

def extract_bot_reply(full_text: str) -> str:
    idx = full_text.find("BOT:")
    if idx == -1:
        return full_text.strip()
    fragment = full_text[idx + len("BOT:") :]
    return fragment.strip()

def main() -> None:
    text = load_corpus()
    print("Training word-level chat model on data/chat_corpus.txt")
    model = WordNGramModel.from_corpus(text, k=3, alpha=0.5)
    print("Type a message. Empty line to exit.")
    while True:
        try:
            user = input("USER: ").strip()
        except EOFError:
            break
        if not user:
            break
        prompt = f"USER: {user}\nBOT:"
        generated = model.generate(prompt, max_tokens=40, seed=None)
        reply = extract_bot_reply(generated)
        print(f"BOT: {reply}")

if __name__ == "__main__":
    main()
