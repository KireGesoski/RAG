from openai import OpenAI
from conf_file import openAi_key

class ImportanceAgent:
    def __init__(self):
        self.client = OpenAI(api_key=openAi_key)

    def analyze_text(self, text: str) -> str:
        """
        Decide if a statement contains personal info, a car search request, or is irrelevant.
        - If personal info -> 'personal'
        - If specific car search -> 'references'
        - Else -> 'No DB'
        """

        prompt = f"""
            You are an assistant that classifies user statements.

            If the text contains:
            - any personal information (like name, phone, location, email, or personal preferences unrelated to cars) → respond exactly: personal
            - details about what kind of car they are searching for (e.g., year, type, fuel, price, mileage, make, model, or 'I'm looking for') → respond exactly: references
            - anything else → respond exactly: No DB

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
    agent = ImportanceAgent()

    samples = [
        "My name is John and I live in Berlin.",
        "I'm looking for a 2015 diesel sedan under 10,000 euros.",
        "The weather is great today.",
        "You can contact me at john@example.com.",
        "I prefer something compact and petrol, maybe from 2012 onwards."
    ]

    for text in samples:
        print(f"Input: {text}")
        print("Output:", agent.analyze_text(text))
        print("-" * 50)
