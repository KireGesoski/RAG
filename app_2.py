import json

from guardrails import Guard
from guardrails.hub import RegexMatch
import os
from llama_cpp import Llama
from openinference.semconv.trace import (
    SpanAttributes as OI,
    OpenInferenceSpanKindValues as Kind,
)
from Experiment import Experiment
from phoenix.session.client import Client

# 1) Connect to your local Phoenix (base URL, no /v1 suffix)
ph = Client(endpoint="http://localhost:6006")  # <- local Phoenix UI/API
existing = ph.get_dataset(name="new_dataset_01")
guard = Guard().use(
    RegexMatch,
    regex=r"^(NO_ANSWER|ANSWER:\s.+)$",
)

from phoenix.otel import register

tracer_provider = register(
    endpoint="http://localhost:6006/v1/traces",   # HTTP collector
    protocol="http/protobuf",
    auto_instrument=False,
    batch=True,
    set_global_tracer_provider=False,
)
# ✅ Use Phoenix's tracer directly
tracer = tracer_provider.get_tracer("llama_cpp")

llm = Llama(
    model_path="models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf",
    n_ctx=32768,
    verbose=False
)

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
dataset_records = []
def normalize_output(text: str) -> str:
    s = text.strip()

    if s.upper().startswith("NO_ANSWER"):
        return "NO_ANSWER"

    if s.upper().startswith("ANSWER:"):
        s = s[len("ANSWER:"):].strip()

    return f"ANSWER: {s}" if s else "NO_ANSWER"

def call_llm_direct(question: str) -> str:
    print('in the model')
    with tracer.start_as_current_span(
        "qa_chain",
        attributes={
            OI.OPENINFERENCE_SPAN_KIND: Kind.CHAIN.value,  # "CHAIN"
            OI.INPUT_VALUE: question,                    # <- Phoenix shows this as Input
        },
    ) as chain_span:

        # (optional) Child span for the model call
        with tracer.start_as_current_span(
            "llama",
            attributes={
                OI.OPENINFERENCE_SPAN_KIND: "LLM",         # child marked as LLM if you like
            },
        ) as llm_span:
            try:
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=120,
                    temperature=0,
                    top_p=0.1,
                    repeat_penalty=1.15,
                    stop=["</s>"]
                )

                text  = resp["choices"][0]["message"]["content"].strip()
                usage = resp.get("usage", {})

                # Optional metrics on child span
                llm_span.set_attribute(OI.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0))
                llm_span.set_attribute(OI.LLM_TOKEN_COUNT_COMPLETION, usage.get("completion_tokens", 0))
                llm_span.set_attribute(OI.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens", 0))
                llm_span.set_attribute("mock_tokens",
                                       usage.get("total_tokens", 0) or 1)

                # Put the FINAL answer on the parent so the chain card shows Input + Output
                chain_span.set_attribute(OI.OUTPUT_VALUE, text)

                validated_text = guard.validate(text)
                experiment = Experiment('exp_demo_clean_5', question, validated_text.raw_llm_output)
                experiment.run()
                return validated_text

            except Exception as e:
                llm_span.record_exception(e)
                chain_span.set_attribute("llm.error", str(e))
                return "NO_ANSWER"

def main():
    try:
        while True:
            q = input("\nQuestion: ").strip()
            if not q:
                continue
            if q == 'q':
                exit(0)
            answer = call_llm_direct(q)
            print(answer)
    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()