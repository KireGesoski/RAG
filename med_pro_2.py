#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import os
import sys
import textwrap


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)

# Point to your LLM
GGUF_PATH = os.path.join(
    SCRIPT_DIR,
    "models",
    "qwen2.5-0.5b-instruct",
    "qwen2.5-0.5b-instruct-q4_0.gguf",
)

# Parameters
N_CTX = 2048
TEMPERATURE = 0.2
MAX_TOKENS = 256

# Data
DOCS: List[Dict[str, str]] = [
    {"id": "monograph.md#intro", "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain, fever, and inflammation."},
    {"id": "monograph.md#common-side-effects", "text": "Common side effects include nausea, dizziness, stomach upset, and heartburn. Taking with food or milk may reduce stomach irritation."},
    {"id": "monograph.md#serious-risks", "text": "Serious risks with high doses or long-term use can include increased risk of heart attack or stroke, and gastrointestinal bleeding or ulcers."},
    {"id": "dosing.txt#adult-otc", "text": "Typical adult OTC dose is 200–400 mg every 4–6 hours as needed. Do not exceed 1200 mg/day OTC without physician supervision."},
    {"id": "warnings.md#contra", "text": "Avoid ibuprofen right before or after heart surgery (CABG). Consult a healthcare professional if you have kidney disease, ulcers, or are on blood thinners."},
    {"id": "interactions.md#other-nsaids", "text": "Avoid combining ibuprofen with other NSAIDs. Check for interactions with anticoagulants and certain antihypertensives."},
]

# Retriever (TF-IDF)
def _sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
        from sklearn.metrics.pairwise import cosine_similarity  # noqa
    except Exception as e:
        print("[ERROR] scikit-learn missing. Install it with:", file=sys.stderr)
        print("  python -m pip install scikit-learn", file=sys.stderr)
        raise

class TfidfRetriever:
    def __init__(self, docs: List[Dict[str, str]]):
        _sklearn()
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = docs
        self.corpus = [d["text"] for d in docs]
        self.ids = [d["id"] for d in docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).ravel()
        top_idx = sims.argsort()[::-1][:k]
        return [{"id": self.ids[i], "text": self.corpus[i], "score": float(sims[i])} for i in top_idx]

# Prompt template
PROMPT_TMPL = """You are a careful medical assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't know from the provided context."

# Context
{context}

# Question
{question}

# Instructions
- Be concise and factual.
- Include brief inline citations like [source:id].

# Answer:
"""

def build_context(hits: List[Dict]) -> str:
    return "\n".join(f"- ({h['id']}, score={h['score']:.3f}) {h['text']}" for h in hits)

# LLM
class LocalLLM:
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self._binding_mode = False
        self._llm = None
        self._init_backend()

    def _init_backend(self, llama_cpp=None):
        if not os.path.exists(self.gguf_path):
            raise FileNotFoundError(f"GGUF model not found at:\n  {self.gguf_path}\n"
                                    f"Make sure you downloaded it to this path.")
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.gguf_path,
                n_ctx=N_CTX,
                verbose=False,
            )
            self._binding_mode = True
            return
        except Exception as e:
            print(f"[WARN] llama-cpp-python not available or failed to load: {e}")
            print("[INFO] Falling back to llama-cli if available.")

    def generate(self, prompt: str, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
        if self._binding_mode:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return out["choices"][0]["message"]["content"].strip()
        return "None"

# RAG
@dataclass
class RagAnswer:
    answer: str
    hits: List[Dict]
    prompt_preview: str

def rag_answer(question: str, llm: LocalLLM, k: int = 3) -> RagAnswer:
    retriever = TfidfRetriever(DOCS)
    hits = retriever.retrieve(question, k=k)
    context = build_context(hits)
    prompt = PROMPT_TMPL.format(context=context, question=question)
    preview = textwrap.shorten(prompt, width=900, placeholder=" …")
    answer = llm.generate(prompt)
    return RagAnswer(answer=answer, hits=hits, prompt_preview=preview)


# Process
def main():
    llm = LocalLLM(GGUF_PATH)
    print("Ask a question or q exit.")
    try:
        while True:
            q = input("\nQuestion: ").strip()
            if not q:
                continue
            if q.lower() == 'q':
                print("Exiting…")
                break
            res = rag_answer(q, llm, k=3)
            print(res.answer)
    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()

"""
Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is commonly used to relieve pain and reduce inflammation....
"""