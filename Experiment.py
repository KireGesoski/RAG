from Dataset import Dataset
from phoenix.experiments import run_experiment
from difflib import SequenceMatcher

class Experiment:
    def __init__(self, dataset_name, question, answer, prediction):
        self.dataset_name = dataset_name
        self.question = question
        self.answer = answer
        self.prediction = prediction

        self.ds = Dataset(dataset_name, question, answer)

    def run(self):
        dataset = self.ds.save()

        def task(input):
            q = (input.get("question") or "").strip().lower()
            print("Question", q)
            if q == self.question.strip().lower():
                return {"prediction": self.prediction}
            return {"prediction": "UNKNOWN"}

        def exact_match(output, expected):
            print("🔍 Expected:", expected)
            print("🔍 Output:", output)
            return 1.0 if output.get("prediction") == expected.get("answer") else 0.0

        def semantic_match(output, expected):
            """
            Returns similarity between 0 and 1 based on text similarity.
            """
            _pred = output["prediction"]
            _exp = expected["answer"]
            score = SequenceMatcher(None, _pred, _exp).ratio()
            return score

        experiment = run_experiment(
            dataset=dataset,
            task=task,
            evaluators=[exact_match],
            experiment_name="initial-experiment",
            print_summary=True,
        )

        print(f"✅ Ran experiment on: '{self.question}' → Prediction: '{self.prediction}'")

    def summary(self):
        print("\n📌 Summary:")
        print(f"Dataset: {self.dataset_name}")
        print(f"Question: {self.question}")
        print(f"Prediction: {self.prediction}")
