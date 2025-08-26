from llama_cpp import Llama
import os

llm = Llama(
    model_path="models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf",
    n_ctx=32768,
    verbose=False
)
SYS = (
  "If you are not certain, respond exactly with: NO_ANSWER\n"
  "If certain, respond with: ANSWER: <concise fact>\n"
  "Do not add anything else."
)

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

    #return resp["choices"][0]["text"].strip()
    text = resp["choices"][0]["message"]["content"].strip()
    print('Text', resp)

    if text.startswith("NO_ANSWER"):
        return 'No Answer Provided'
    return text
# resp object {'id': 'cmpl-2b9e228e-d4db-4193-82bd-c42974ed1599', 'object': 'text_completion', 'created': 1755685239, 'model': 'models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf', 'choices': [{'text': ', and what are its benefits? Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, inflammation, and fever. It is also used to relieve pain and reduce swelling in the eyes. Ibuprofen is commonly used to relieve pain and reduce swelling in the eyes, such as when eye irritation is present. It is also used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 256, 'total_tokens': 261}}

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
"""
=== LLM Answer ===
The use of ibuprofen is increasing in the United States. The most common type of ibuprofen used is ibuprofen 400mg. Ibuprofen is a non-steroidal anti-inflammatory drug. Non-steroidal anti-inflammatory drugs are used to relieve pain and reduce inflammation. The most common use of ibuprofen is as a pain reliever.
Ibuprofen is used to reduce inflammation and relieve pain. Ibuprofen is used to relieve pain and reduce inflammation. The most common use of ibuprofen is as a pain reliever. Ibuprofen 400mg is a non-steroidal anti-inflammatory drug. The most common use of ibuprofen is as a pain reliever. The most common use of ibuprofen is as a pain reliever. The most common use of ibuprofen is as a pain reliever. The most common use of ibuprofen is a pain reliever. The most common use of ibuprofen is a pain reliever.
What are the effects of Ibuprofen? Ibuprofen can be used to relieve pain. Ibuprofen can be used to relieve pain. Ibuprofen can be used to relieve pain. The most common use of ibuprofen is a
"""