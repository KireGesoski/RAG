from llama_cpp import Llama

class SimpleLLM:
    def __init__(self):
        self.llm = Llama(
            model_path="models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf",
            n_ctx=32768,
            verbose=False
        )
        self.sys = (
            "If you are not certain, respond exactly with: NO_ANSWER\n"
            "If certain, respond with: ANSWER: <concise fact>\n"
            "Do not add anything else."
        )

    def call_llm_direct(self, question: str) -> str:
        print('in the model')
        resp = self.llm.create_chat_completion(
             messages=[
                 {"role": "system", "content": self.sys},
                 {"role": "user", "content": question}
             ],
             max_tokens=120,
             temperature=0.2,
             top_p=0.9,
             repeat_penalty=1.15,
             stop=["</s>"]
        )
        # return resp["choices"][0]["text"].strip()
        text = resp["choices"][0]["message"]["content"].strip()
        print('Text', resp)

        if text.startswith("NO_ANSWER"):
            return 'No Answer Provided'
        return text