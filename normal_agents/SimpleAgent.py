from openai import OpenAI
from conf_file import openAi_key

class SimpleAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=openAi_key)
        self.model = model

        # System message for consistent behavior
        self.sys = (
            "If you are not certain, respond exactly with: NO_ANSWER\n"
            "If certain, respond with: ANSWER: <concise fact>\n"
            "Do not add anything else."
        )

    def call_llm_direct(self, question: str) -> str:
        print("in the OpenAI model")

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.sys},
                    {"role": "user", "content": question}
                ],
                max_tokens=120,
                temperature=0.2,
                top_p=0.9,
            )

            text = resp.choices[0].message.content.strip()
            print("Text:", text)

            if text.startswith("NO_ANSWER"):
                return "No Answer Provided"
            return text

        except Exception as e:
            print("❌ OpenAI API error:", e)
            return "Error calling model"
