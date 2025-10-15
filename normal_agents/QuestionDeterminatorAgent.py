from openai import OpenAI
from conf_file import openAi_key


class QuestionDeterminationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=openAi_key)

    def analyze_text(self, text: str) -> str:
        """
        If the text is a question about the car market or buying cars, return the question.
        Otherwise, return None.
        """
        prompt = f"""
            You are an assistant that determines whether a text is a QUESTION about the car market,
            buying/selling cars, or anything automobile-related.

            If it is, return the exact same question text.
            If it is not, return exactly this sentence: None.

            Text: "{text}"
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    agent = QuestionDeterminationAgent()

    samples = [
        "Is now a good time to buy a new car?",
        "What are Tesla’s Q3 earnings?",
        "Tell me about Python decorators.",
        "Should I sell my used Toyota before winter?"
    ]

    for text in samples:
        print(f"Input: {text}")
        print("Output:", agent.analyze_text(text))
        print("-" * 50)
