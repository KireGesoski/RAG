import os
import pandas as pd
from phoenix.session.client import Client
from typing import List, Dict, Optional

class Dataset:
    def __init__(
        self,
        dataset_name: str,
        questions: List[str],
        answers: List[str],
        endpoint: Optional[str] = None
    ):
        """
        Phoenix dataset helper for multiple questions/answers.

        Args:
            dataset_name: Name of the Phoenix dataset.
            questions: List of questions (must match length of answers).
            answers: List of expected answers (must match length of questions).
            endpoint: Phoenix base URL (defaults to $PHOENIX_ENDPOINT or http://localhost:6006)
        """
        if len(questions) != len(answers):
            raise ValueError("Length of questions and answers must match")

        self.dataset_name = dataset_name
        self.rows: List[Dict[str, str]] = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        self.endpoint = endpoint or os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
        self._client = Client(endpoint=self.endpoint)

    def add(self, question: str, answer: str) -> None:
        """Add a new question/answer pair to the dataset."""
        self.rows.append({"question": question, "answer": answer})

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert internal rows to a pandas DataFrame."""
        return pd.DataFrame(self.rows, columns=["question", "answer"])

    def save(self):
        """
        Uploads a new dataset if it doesn't exist, otherwise appends rows.
        Prints the Phoenix UI URL at the end.
        """
        df = self._to_dataframe()
        try:
            dataset = self._client.get_dataset(name=self.dataset_name)
            print("Dataset found — appending rows...")
            self._client.append_to_dataset(
                dataset_name=self.dataset_name,
                dataframe=df,
                input_keys=["question"],
                output_keys=["answer"],
            )
        except Exception:
            print("Creating dataset and uploading rows...")
            dataset = self._client.upload_dataset(
                dataset_name=self.dataset_name,
                dataframe=df,
                input_keys=["question"],
                output_keys=["answer"],
                dataset_description=f"Local dataset '{self.dataset_name}'",
            )
        print("Open in UI:", self._client.web_url)
        return dataset
