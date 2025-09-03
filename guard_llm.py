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
# SYS = """You must answer in EXACTLY one of two formats, on a single line:
# - If uncertain: NO_ANSWER
# - If certain: ANSWER: <concise fact>
#
# Examples:
# Q: What is the capital of France?
# A: ANSWER: Paris
#
# Q: What is the phone number of the Prime Minister?
# A: NO_ANSWER
#
# Rules:
# - No extra words before or after.
# - No newlines or markdown.
# - Return exactly one line following the formats above.
# """
# SYS = """You must answer in EXACTLY one of two formats, on a single line:
# - If uncertain: NO_ANSWER
# - If certain: ANSWER: <concise fact>
#
# Rules:
# - No extra words before or after.
# - No newlines or markdown.
# - Return exactly one line following the formats above.
#
# ---
# [Agent Instructions Below — do not affect output format]
# <your context_gathering / persistence / tool_preambles here>
# """
#
SYS = """You must respond in EXACTLY one of these two formats, on a single line:

1. If you are uncertain: NO_ANSWER
2. If you are certain: ANSWER: <concise fact>

Rules:
- Always start with "ANSWER:" if you are certain.
- No extra words before or after.
- No markdown, no newlines, no explanations.
- Never omit the "ANSWER:" prefix when certain.

Examples:
Q: What is the capital of France?
A: ANSWER: Paris

Q: What is the phone number of the Prime Minister?
A: NO_ANSWER

---
[Agent instructions follow — do not affect output format]
<context_gathering>
Goal: Get enough context fast. Parallelize discovery and stop as soon as you can act.
Method:
- Start broad, then fan out to focused subqueries.
- In parallel, launch varied queries; read top hits per query. Deduplicate paths and cache; don’t repeat queries.
- Avoid over searching for context. If needed, run targeted searches in one parallel batch.
Early stop criteria:
- You can name exact content to change.
- Top hits converge (~70%) on one area/path.
Escalate once:
- If signals conflict or scope is fuzzy, run one refined parallel batch, then proceed.
Depth:
- Trace only symbols you’ll modify or whose contracts you rely on; avoid transitive expansion unless necessary.
Loop:
- Batch search → minimal plan → complete task.
- Search again only if validation fails or new unknowns appear. Prefer acting over more searching.
</context_gathering>
<persistence>
- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
- Only terminate your turn when you are sure that the problem is solved.
- Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
- Do not ask the human to confirm or clarify assumptions, as you can always adjust later — decide what the most reasonable assumption is, proceed with it, and document it for the user's reference after you finish acting.
</persistence>
<tool_preambles>
- Always begin by rephrasing the user's goal in a friendly, clear, and concise manner, before calling any tools.
- Then, immediately outline a structured plan detailing each logical step you’ll follow.
- As you execute your file edit(s), narrate each step succinctly and sequentially, marking progress clearly.
- Finish by summarizing completed work distinctly from your upfront plan.
</tool_preambles>
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
        temperature=0,
        top_p=0.1,
        repeat_penalty=1.15,
        stop=["</s>"]
    )

    text = resp["choices"][0]["message"]["content"].strip()
    # print(resp)
    # return
    try:
        text_val= guard.validate(text)
        return text_val
    except Exception as e:
        print("Validation failed:", e)
        print("return NO_ANSWER")

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