#from langmem import LangMem, InMemoryStore

from openai import OpenAI
# import langmem
# print(dir(langmem))

# Initialize OpenAI client
#client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# # Create memory store
# store = InMemoryStore(
#     index={
#         "dims": 1536,  # size of embedding vectors
#         "embed": "openai:text-embedding-3-small",  # which embedding model to use
#     }
# )
#
# # Initialize LangMem with memory store + LLM
# mem = LangMem(
#     store=store,
#     client=client,
#     model="gpt-4o-mini",  # can be gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.
# )
#
# # Function to ask questions to the LLM with memory
# def ask(question: str):
#     answer = mem(question)
#     print(f"Q: {question}")
#     print(f"A: {answer}\n")
#
# # First interaction
# ask("My name is Alice.")
#
# # Ask again later without repeating the name
# ask("What is my name?")
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LangMem:
    def __init__(self):
        self.memories: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def add_memory(self, text: str):
        """Add new memory and update TF-IDF matrix."""
        self.memories.append(text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.memories)

    def search_memory(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant memories for a query using cosine similarity."""
        if not self.memories:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idxs = scores.argsort()[::-1][:top_k]
        return [self.memories[i] for i in top_idxs if scores[i] > 0]

    def update_memory(self, old_text: str, new_text: str):
        """Update memory by replacing old with new."""
        for i, txt in enumerate(self.memories):
            if txt == old_text:
                self.memories[i] = new_text
                self.tfidf_matrix = self.vectorizer.fit_transform(self.memories)
                break

    def __call__(self, query: str) -> List[str]:
        """Shortcut: search memory when called."""
        return self.search_memory(query)
