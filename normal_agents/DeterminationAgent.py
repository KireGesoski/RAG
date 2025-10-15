from openai import OpenAI

from conf_file import openAi_key

class DeterminationAgent:
    """
    A simple agent that determines whether a given text is a Question or a Statement.
    """

    def __init__(self, api_key: str = openAi_key, model: str = "gpt-4o-mini"):
        """
        Initialize the agent with an OpenAI client.
        :param api_key: OpenAI API key.
        :param model: Model name (default: gpt-4o-mini).
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def classify_text(self, text: str) -> str:
        """
        Determines whether the input text is a question or a statement.

        :param text: The input text to analyze.
        :return: 'Question' or 'Statement'.
        """
        prompt = (
            "You are a classification agent. "
            "Read the following text and respond with only one word: "
            "'Question' if the text is asking for information, or 'Statement' if it is a declarative sentence.\n\n"
            f"Text: {text}\n\n"
            "Answer:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful text classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # Normalize output just in case model adds formatting
        if "question" in result.lower():
            return "Question"
        elif "statement" in result.lower():
            return "Statement"
        else:
            return "Unknown"

# Example usage:
if __name__ == "__main__":
    agent = DeterminationAgent(api_key=openAi_key)
    text_samples = [
        "What is the capital of France?",
        "The capital of France is Paris."
    ]

    for text in text_samples:
        print(f"Text: {text}")
        print(f"Type: {agent.classify_text(text)}\n")
