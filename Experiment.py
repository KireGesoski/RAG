from Dataset import Dataset
from phoenix.experiments import run_experiment
from difflib import SequenceMatcher
from LLMJudge import LLMJudgeModel

class Experiment:
    def __init__(self, dataset_name, questions, answers, predictions):
        self.dataset_name = dataset_name
        self.questions = questions
        self.answers = answers
        self.predictions = predictions

        self.ds = Dataset(dataset_name, questions, answers)

    def run(self):
        dataset = self.ds.save()

        # def task(input):
        #     q = (input.get("question") or "").strip().lower()
        #     print("Question", q)
        #     if q == self.question.strip().lower():
        #         return {"prediction": self.prediction}
        #     return {"prediction": "UNKNOWN"}

        def task(input):
            q = (input.get("question") or "").strip().lower()
            # Find index of matching question
            try:
                idx = [question.lower() for question in self.questions].index(q)
                return {"prediction": self.predictions[idx]}
            except ValueError:
                return {"prediction": "UNKNOWN"}

        def exact_match(output, expected):
            # print("🔍 Expected:", expected)
            # print("🔍 Output:", output)
            return 1.0 if output.get("prediction") == expected.get("answer") else 0.0

        def semantic_match(output, expected):
            """
            Returns similarity between 0 and 1 based on text similarity.
            """
            _pred = output["prediction"]
            _exp = expected["answer"]
            score = SequenceMatcher(None, _pred, _exp).ratio()
            return score

        # def llm_judge_match(outputs):
        #     print('OUTPUTS:', outputs)
        #     print('QUESTIONS:', self.questions)
        #     llm_judge = LLMJudgeModel()
        #     scores = []
        #     for output, question in zip(outputs, self.questions):
        #         judge_answer = llm_judge.get_answer(question)
        #         print('Prediction', output["prediction"])
        #         print('Judge Answer', judge_answer)
        #         score = llm_judge.semantic_match(output["prediction"], judge_answer, binary=False)
        #         scores.append(score)
        #     return scores
        def llm_judge_match(output):
            llm_judge = LLMJudgeModel()
            judge_answer = llm_judge.get_answer(self.questions[0])

            print("Prediction:", output["prediction"])
            print("Judge Answer:", judge_answer)

            score = llm_judge.semantic_match(output["prediction"], judge_answer, binary=False)
            return score

        experiment = run_experiment(
            dataset=dataset,
            task=task,
            evaluators=[semantic_match, llm_judge_match],
            experiment_name="initial-experiment",
            print_summary=True,
        )

        print(f"✅ Ran experiment on: '{self.questions}' → Prediction: '{self.predictions}'")

    def summary(self):
        print("\n📌 Summary:")
        print(f"Dataset: {self.dataset_name}")
        print(f"Question: {self.question}")
        print(f"Prediction: {self.prediction}")
