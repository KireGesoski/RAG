# from langgraph.prebuilt import create_react_agent
# from langgraph.store.memory import InMemoryStore
# from langmem import create_manage_memory_tool, create_search_memory_tool
# from conf_file import openAi_key
# import os
#
# # ✅ Set OpenAI API key globally
# os.environ["OPENAI_API_KEY"] = openAi_key
#
# # Set up storage
# store = InMemoryStore(
#     index={
#         "dims": 1536,
#         "embed": "openai:text-embedding-3-small",
#     }
# )
#
# # Create agent
# agent = create_react_agent(
#     "anthropic:claude-3-5-sonnet-latest",
#     tools=[
#         create_manage_memory_tool(namespace=("memories",)),
#         create_search_memory_tool(namespace=("memories",)),
#     ],
#     store=store,
# )
#
# # Invoke agent
# #print(agent)
# agent.invoke({
#     "messages": [
#         {"role": "user", "content": "Remember that I prefer dark mode."}
#     ]
# })
from typing import List, Dict, Callable, Any, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LangMemHybrid:
    def __init__(self):
        # --- Semantic memory ---
        self.semantic_memories: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

        # --- Episodic memory ---
        self.episodic_memories: List[Dict[str, Any]] = []  # stores {event, timestamp, context}

        # --- Procedural memory ---
        self.skills: Dict[str, Callable] = {}  # skill name -> function

    # ---------------- SEMANTIC ----------------
    def add_semantic(self, text: str):
        """Add factual/knowledge memory."""
        self.semantic_memories.append(text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.semantic_memories)

    def search_semantic(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant semantic memories."""
        if not self.semantic_memories:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idxs = scores.argsort()[::-1][:top_k]
        return [self.semantic_memories[i] for i in top_idxs if scores[i] > 0]

    # ---------------- EPISODIC ----------------
    def add_episode(self, event: str, context: Optional[str] = None):
        """Add episodic memory with timestamp + optional context."""
        self.episodic_memories.append({
            "event": event,
            "timestamp": datetime.now(),
            "context": context
        })

    def recall_episodes(self, n: int = 3) -> List[Dict[str, Any]]:
        """Recall the n most recent episodes."""
        return self.episodic_memories[-n:]

    # ---------------- PROCEDURAL ----------------
    def add_skill(self, name: str, func: Callable):
        """Store a callable skill."""
        self.skills[name] = func

    def run_skill(self, name: str, *args, **kwargs) -> Any:
        """Execute a stored skill by name."""
        if name not in self.skills:
            raise ValueError(f"Skill '{name}' not found.")
        return self.skills[name](*args, **kwargs)

    # ---------------- UNIVERSAL CALL ----------------
    def __call__(self, query: str) -> Dict[str, Any]:
        """Unified interface: search semantic + recall episodes."""
        return {
            "semantic": self.search_semantic(query),
            "episodes": self.recall_episodes(),
            "skills": list(self.skills.keys())
        }
