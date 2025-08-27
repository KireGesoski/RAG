from guardrails import Guard
from guardrails.hub import RegexMatch
import os
from llama_cpp import Llama

guard = Guard().use(
    RegexMatch,
    regex=r"^(NO_ANSWER|ANSWER:\s.+)$",
)

llm = Llama(
    model_path="models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf",
    n_ctx=32768,
    verbose=False
)
SYS = """You must answer in EXACTLY one of two formats, on a single line:
- If uncertain: NO_ANSWER
- If certain: ANSWER: <concise fact>

Examples:
Q: What is the capital of France?
A: ANSWER: Paris

Q: What is the phone number of the Prime Minister?
A: NO_ANSWER

Rules:
- No extra words before or after.
- No newlines or markdown.
- Return exactly one line following the formats above.
"""
def normalize_output(text: str) -> str:
    s = text.strip()

    if s.upper().startswith("NO_ANSWER"):
        return "NO_ANSWER"

    if s.upper().startswith("ANSWER:"):
        s = s[len("ANSWER:"):].strip()

    return f"ANSWER: {s}" if s else "NO_ANSWER"

def call_llm_direct(question: str) -> str:
    print('in the model')
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": question}
        ],
        max_tokens=120,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["</s>"]
    )

    text = resp["choices"][0]["message"]["content"].strip()

    try:
        text_val= guard.validate(normalize_output(text))
        return text_val
    except Exception as e:
        print("Validation failed:", e)
        print("-> return NO_ANSWER")

def main():
    try:
        while True:
            q = input("\nQuestion: ").strip()
            if not q:
                continue
            answer = call_llm_direct(q)
            print(answer)
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()