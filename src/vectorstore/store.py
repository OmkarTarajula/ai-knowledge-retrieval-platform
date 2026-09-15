import os
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStoreManager:
    """Manages document embeddings, persistence, and cosine similarity search without DLL dependencies."""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.data_file = os.path.join(persist_directory, "vector_data.json")
        os.makedirs(persist_directory, exist_ok=True)
        
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        
        self._load()

    def _load(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                if self.chunks:
                    corpus = [c["text"] for c in self.chunks]
                    self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            except Exception:
                self.chunks = []

    def _save(self):
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> int:
        if not new_chunks:
            return 0

        existing_ids = {c["chunk_id"] for c in self.chunks}
        added = 0
        for chunk in new_chunks:
            if chunk["chunk_id"] not in existing_ids:
                self.chunks.append(chunk)
                existing_ids.add(chunk["chunk_id"])
                added += 1

        if self.chunks:
            corpus = [c["text"] for c in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self._save()

        return added

    def query(self, query_text: str, top_k: int = 3, domain_filter: str = None) -> List[Dict[str, Any]]:
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Filter candidate pool by domain if specified
        if domain_filter and domain_filter != "General":
            candidate_indices = [
                i for i, c in enumerate(self.chunks)
                if c.get("metadata", {}).get("domain") in [domain_filter, "General"]
            ]
        else:
            candidate_indices = list(range(len(self.chunks)))

        if not candidate_indices:
            return []

        # Calculate cosine similarity vector
        query_vec = self.vectorizer.transform([query_text])
        sub_matrix = self.tfidf_matrix[candidate_indices]
        similarities = cosine_similarity(query_vec, sub_matrix).flatten()

        # Rank and pick top-K
        top_sub_idx = np.argsort(similarities)[::-1][:min(top_k, len(candidate_indices))]

        results = []
        for idx in top_sub_idx:
            orig_idx = candidate_indices[idx]
            score = float(similarities[idx])
            chunk = self.chunks[orig_idx]
            results.append({
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
                "metadata": chunk.get("metadata", {}),
                "similarity_score": round(score, 4)
            })

        return results