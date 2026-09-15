from typing import Dict, Any, List

class ResponseGenerationAgent:
    """
    Generates grounded answers strictly from retrieved context chunks.
    Adapts response structure to query type (factual, procedural, comparative)
    and includes source metadata attribution and confidence indicators.
    """

    def generate_response(self, query: str, query_analysis: Dict[str, Any], retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        query_type = query_analysis.get("query_type", "factual")
        status = retrieval_result.get("status", "empty")
        chunks: List[Dict[str, Any]] = retrieval_result.get("chunks", [])

        # 1. Fallback if no context was found or low confidence
        if status in ["empty", "low_confidence"] or not chunks:
            return {
                "answer": "The available knowledge base does not maintain sufficient information to answer your query. Please refer to an official administrator or upload relevant documents.",
                "sources": [],
                "confidence_score": 0.20,
                "confidence_level": "Low",
                "query_type": query_type,
                "grounded": False
            }

        # 2. Extract citations/metadata
        sources = []
        for c in chunks:
            meta = c.get("metadata", {})
            sources.append({
                "file_name": meta.get("file_name", "Unknown"),
                "page_number": meta.get("page_number", 1),
                "section": meta.get("section", "General"),
                "chunk_id": c.get("chunk_id", ""),
                "similarity_score": c.get("similarity_score", 0.0)
            })

        # Calculate overall system confidence
        top_score = retrieval_result.get("top_similarity", 0.0)
        confidence_level = "High" if top_score >= 0.75 else "Medium"

        # 3. Format grounded response based on query type
        primary_text = chunks[0]["text"]
        context_summary = "\n".join([f"- {c['text']}" for c in chunks[:2]])

        if query_type == "procedural":
            formatted_answer = (
                f"Based on the documented procedure, follow these steps:\n\n"
                f"{primary_text}\n\n"
                f"*(Note: Follow all sequence steps as outlined in the reference documents.)*"
            )
        elif query_type == "comparative":
            formatted_answer = (
                f"Comparison details extracted from verified documentation:\n\n"
                f"{context_summary}\n\n"
                f"*(Summary: Verify specific criteria against each section above.)*"
            )
        else:  # factual default
            formatted_answer = (
                f"According to verified documentation:\n\n"
                f"{primary_text}"
            )

        return {
            "answer": formatted_answer,
            "sources": sources,
            "confidence_score": top_score,
            "confidence_level": confidence_level,
            "query_type": query_type,
            "grounded": True
        }