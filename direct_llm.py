from llama_cpp import Llama

# Load the model once
llm = Llama(model_path="models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf", verbose=False)

def call_llm_direct(question: str) -> str:
    """Send the question straight to the LLM (no RAG)."""
    resp = llm(
        question,
        max_tokens=256,
        stop=["</s>"],  # optional, depends on the model
    )
    print(resp)
    return
    return resp["choices"][0]["text"].strip()
# resp object {'id': 'cmpl-2b9e228e-d4db-4193-82bd-c42974ed1599', 'object': 'text_completion', 'created': 1755685239, 'model': 'models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf', 'choices': [{'text': ', and what are its benefits? Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that is used to treat pain, inflammation, and fever. It is also used to relieve pain and reduce swelling in the eyes. Ibuprofen is commonly used to relieve pain and reduce swelling in the eyes, such as when eye irritation is present. It is also used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain and reduce swelling in the knee. Ibuprofen is commonly used to relieve pain', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 256, 'total_tokens': 261}}

def main():
    # print("🧠 Direct LLM mode (no RAG). Ask a question, or Ctrl+C to exit.")
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