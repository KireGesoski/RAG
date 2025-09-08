import os
import pandas as pd
from phoenix.session.client import Client

class Dataset:
    def __init__(self, dataset_name: str, question: str, answer: str, endpoint: str | None = None):
        """
        Create a dataset helper tied to a Phoenix server.

        Args:
            dataset_name: Name of the Phoenix dataset.
            question: First input row's question.
            prediction: First input row's prediction.
            endpoint: Phoenix base URL (defaults to $PHOENIX_ENDPOINT or http://localhost:6006).
        """
        self.dataset_name = dataset_name
        self.rows = [{"question": question, "answer": answer}]
        self.endpoint = endpoint or os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
        self._client = Client(endpoint=self.endpoint)

    # def add(self, question: str, prediction: str) -> None:
    #     """Queue another row to upload/append."""
    #     self.rows.append({"question": question, "prediction": prediction})

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["question", "answer"])

    def save(self):
        """
        Uploads a new dataset if it doesn't exist, otherwise appends.
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
