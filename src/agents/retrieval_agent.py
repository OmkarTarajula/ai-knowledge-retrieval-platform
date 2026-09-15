from typing import List, Dict, Any
from src.vectorstore.store import VectorStoreManager

class RetrievalAgent:
    """
    Wraps vector search with configurable Top-K, similarity thresholds,
    low-confidence filtering, and metadata preservation.
    """

    def __init__(self, vector_store: VectorStoreManager, default_top_k: int = 3, min_similarity_threshold: float = 0.50):
        self.vector_store = vector_store
        self.top_k = default_top_k
        self.min_similarity_threshold = min_similarity_threshold

    def retrieve(
        self, query: str, top_k: int = None, threshold: float = None, domain: str = None
    ) -> Dict[str, Any]:
        """
        Retrieves relevant document chunks and applies confidence threshold filtering.
        """
        k = top_k if top_k is not None else self.top_k
        min_thresh = threshold if threshold is not None else self.min_similarity_threshold

        # 1. Fetch raw matches from vector store
        raw_results = self.vector_store.query(query_text=query, top_k=k, domain_filter=domain)

        if not raw_results:
            return {
                "status": "empty",
                "message": "No relevant documents found in knowledge base.",
                "chunks": [],
                "top_similarity": 0.0
            }

        # 2. Filter matches below similarity threshold
        filtered_chunks: List[Dict[str, Any]] = []
        for match in raw_results:
            score = match.get("similarity_score", 0.0)
            if score >= min_thresh:
                filtered_chunks.append({
                    "chunk_id": match.get("chunk_id"),
                    "text": match.get("text"),
                    "metadata": match.get("metadata", {}),
                    "similarity_score": score
                })

        # 3. Handle cases where all results fall below threshold
        if not filtered_chunks:
            return {
                "status": "low_confidence",
                "message": "Results found, but all fell below the minimum similarity threshold.",
                "chunks": [],
                "top_similarity": raw_results[0].get("similarity_score", 0.0)
            }

        return {
            "status": "success",
            "message": f"Successfully retrieved {len(filtered_chunks)} relevant chunk(s).",
            "chunks": filtered_chunks,
            "top_similarity": filtered_chunks[0].get("similarity_score", 0.0)
        }