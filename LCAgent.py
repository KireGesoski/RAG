from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
import json

class LSAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4", verbose: bool = False):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=api_key,
        )
        self.verbose = verbose
        self.agent = self._create_agent()

    def _create_agent(self):
        """Creates the LangChain agent with classification logic."""

        def classify_text(input_text: str) -> str:
            prompt = f"""
            You are an AI classification agent.

            Given this text:
            "{input_text}"

            Decide and output JSON with:
            - "type": one of ["question", "statement", "important_info", "not_important"]
            - "save_to_db": true or false
            - "reason": a short 1-sentence explanation

            Rules:
            - Questions end with '?' or request information.
            - Important info contains user data, preferences, goals, or facts.
            - Not important are greetings or small talk.
            """

            resp = self.llm.invoke(prompt)
            return resp.content

        tools = [
            Tool(
                name="TextClassifier",
                func=classify_text,
                description="Classifies text and decides if it should be saved to DB",
            )
        ]
        return initialize_agent(
            tools,
            self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
        )

    def classify(self, text: str) -> dict:
        """Classifies the given text and returns structured result."""
        try:
            result_str = self.agent.run(f"Classify this text: {text}")
            # Try to parse JSON result
            result_json = json.loads(result_str)
            return result_json
        except Exception as e:
            if self.verbose:
                print("⚠️ Error parsing agent output:", e)
                print("Raw output:", result_str)
            # fallback classification
            return {
                "type": "unknown",
                "save_to_db": False,
                "reason": "Parsing error or unexpected output.",
            }

if __name__ == "__main__":
    from conf_file import openAi_key  # make sure this exists

    agent = LSAgent(api_key=openAi_key, verbose=True)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent.classify(user_input)
        print("\n🧩 Classification Result:")
        print(json.dumps(result, indent=2))

        if result["save_to_db"]:
            print("💾 → Will be saved to database.")
        else:
            print("🚫 → Ignored as non-important.")

#https://python.langchain.com/docs/integrations/chat/openai/