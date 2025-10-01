from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from conf_file import openAi_key
# Choose embedding provider
# Option A: OpenAI
os.environ["OPENAI_API_KEY"] = openAi_key
from openai import OpenAI
openai_client = OpenAI()   # requires OPENAI_API_KEY in env


class LangMemHybridEmbedding:
    def __init__(self, provider: str = "openai"):
        # --- Semantic memory ---
        self.semantic_memories: List[str] = []
        self.semantic_embeddings: List[np.ndarray] = []
        self.provider = provider

        # --- Episodic memory ---
        self.episodic_memories: List[Dict[str, Any]] = []

        # --- Procedural memory ---
        self.skills: Dict[str, Callable] = {}

    # -------- SEMANTIC --------
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding vector from text using chosen provider."""
        if self.provider == "openai":
            resp = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        else:
            raise ValueError("Unknown provider")

    def add_semantic(self, text: str):
        """Add semantic memory + embedding."""
        self.semantic_memories.append(text)
        self.semantic_embeddings.append(self._embed(text))

    def search_semantic(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve semantically the closest memories."""
        if not self.semantic_memories:
            return []
        query_vec = self._embed(query).reshape(1, -1)
        mem_matrix = np.vstack(self.semantic_embeddings)
        scores = cosine_similarity(query_vec, mem_matrix).flatten()
        top_idxs = scores.argsort()[::-1][:top_k]
        return [self.semantic_memories[i] for i in top_idxs if scores[i] > 0.1]

    # -------- EPISODIC --------
    def add_episode(self, event: str, context: Optional[str] = None):
        self.episodic_memories.append({
            "event": event,
            "timestamp": datetime.now(),
            "context": context
        })

    def recall_episodes(self, n: int = 3) -> List[Dict[str, Any]]:
        return self.episodic_memories[-n:]

    # -------- PROCEDURAL --------
    def add_skill(self, name: str, func: Callable):
        self.skills[name] = func

    def run_skill(self, name: str, *args, **kwargs) -> Any:
        if name not in self.skills:
            raise ValueError(f"Skill '{name}' not found.")
        return self.skills[name](*args, **kwargs)

    def classify_memory(self, text: str) -> str:
        if "I can" in text or "I know how" in text:
            return "skill"
        elif any(t in text for t in ["yesterday", "last week", "today", "I did", "I tried"]):
            return "episodic"
        elif any(t in text for t in ["is", "supports", "means", "refers to"]):
            return "semantic"
        else:
            return "semantic"  # default fallback

    # -------- UNIFIED --------
    def __call__(self, query: str) -> Dict[str, Any]:
        return {
            "semantic": self.search_semantic(query),
            "episodes": self.recall_episodes(),
            "skills": list(self.skills.keys())
        }
