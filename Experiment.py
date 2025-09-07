from Dataset import Dataset
from phoenix.experiments import run_experiment

class Experiment:
    def __init__(self, dataset_name, question, prediction):
        self.dataset_name = dataset_name
        self.question = question
        self.prediction = prediction

        self.ds = Dataset(dataset_name, question, prediction)

    def run(self):
        dataset = self.ds.save()

        def task(input):
            q = (input.get("question") or "").strip().lower()
            if q == self.question.strip().lower():
                return {"prediction": self.prediction}
            return {"prediction": "UNKNOWN"}

        def exact_match(output, expected):
            print("🔍 Expected:", expected)
            print("🔍 Output:", output)
            return 1.0 if output.get("prediction") == expected.get("answer") else 0.0


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
