from openai import OpenAI
from conf_file import openAi_key

class LLMJudgeModel:
    def __init__(self, api_key=openAi_key, model_name="gpt-4", judge_model_name="gpt-4"):
        """
        Initializes the OpenAI client and sets model names.
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name             # model used to answer questions
        self.judge_model_name = judge_model_name # model used to judge similarity

    def get_answer(self, question: str) -> str:
        """
        Returns an answer from OpenAI for a given question.
        """
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[{"role": "user", "content": question}]
        )
        return resp.choices[0].message.content.strip()

    def semantic_match(self, answer1: str, answer2: str, binary: bool = False):
        """
        Compares two answers using OpenAI as a judge.
        If binary=True, returns 0/1 (match or not). Otherwise returns a float 0-1 similarity.
        """
        if binary:
            prompt = f"""
            You are a strict judge. Compare these two answers:

            Answer 1: "{answer1}"
            Answer 2: "{answer2}"

            If they mean the same thing, respond with 1. If they are different, respond with 0.
            Only output 1 or 0, nothing else.
            """
        else:
            prompt = f"""
            You are a strict judge. Compare these two answers:

            Answer 1: "{answer1}"
            Answer 2: "{answer2}"

            Return a number between 0 and 1 representing semantic similarity (1 = identical meaning, 0 = completely different).
            Only output the number.
            """

        resp = self.client.chat.completions.create(
            model=self.judge_model_name,
            temperature=0,
            messages=[{"role": "system", "content": prompt}]
        )

        try:
            value = resp.choices[0].message.content.strip()
            if binary:
                return int(value)
            else:
                return max(0.0, min(1.0, float(value)))  # clamp to [0,1]
        except:
            return 0 if binary else 0.0
