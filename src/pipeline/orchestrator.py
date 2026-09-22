from typing import Dict, Any
from src.agents.query_understanding_agent import QueryUnderstandingAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.response_generation_agent import ResponseGenerationAgent
from src.vectorstore.store import VectorStoreManager

class MultiAgentOrchestrator:
    """
    Central manager coordinating the sequential flow across all three agents:
    Query Understanding -> Retrieval Agent -> Response Generation Agent.
    """

    def __init__(self, vector_store: VectorStoreManager, default_top_k: int = 3, threshold: float = 0.50):
        self.query_agent = QueryUnderstandingAgent()
        self.retrieval_agent = RetrievalAgent(
            vector_store=vector_store,
            default_top_k=default_top_k,
            min_similarity_threshold=threshold
        )
        self.response_agent = ResponseGenerationAgent()

    def process_query(self, user_query: str, domain: str = None, top_k: int = None, threshold: float = None) -> Dict[str, Any]:
        """
        Executes the end-to-end multi-agent resolution pipeline.
        """
        # Step 1: Query Understanding & Classification
        query_analysis = self.query_agent.analyze_query(user_query)

        # Check for ambiguity routing (flagged for Milestone 3 clarification)
        if query_analysis.get("routing") == "clarification":
            clarification_msg = "Your query appears ambiguous or underspecified. Could you please specify which exact policy, category, or procedure you want details on?"
            return {
                "query": user_query,
                "status": "clarification_needed",
                "query_analysis": query_analysis,
                "answer": clarification_msg,
                "sources": [],
                "retrieved_chunks": [],
                "confidence_score": query_analysis.get("confidence", 0.5),
                "confidence_level": "Low",
                "response": {
                    "answer": clarification_msg,
                    "sources": [],
                    "confidence_score": query_analysis.get("confidence", 0.5),
                    "confidence_level": "Low"
                }
            }

        # Step 2: Semantic Retrieval with Confidence Thresholding
        retrieval_result = self.retrieval_agent.retrieve(
            query=user_query,
            top_k=top_k,
            threshold=threshold,
            domain=domain
        )

        # Step 3: Grounded Response Generation with Citations
        generation_result = self.response_agent.generate_response(
            query=user_query,
            query_analysis=query_analysis,
            retrieval_result=retrieval_result
        )

        retrieved_chunks = retrieval_result.get("chunks", [])
        sources = generation_result.get("sources", [])
        answer = generation_result.get("answer", "")
        confidence_score = generation_result.get("confidence_score", 0.0)
        confidence_level = generation_result.get("confidence_level", "Low")

        return {
            "query": user_query,
            "status": "success",
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "query_analysis": query_analysis,
            "retrieval_result": retrieval_result,
            "response": generation_result
        }

    def run(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Convenience alias for process_query."""
        return self.process_query(user_query, **kwargs)

    def __call__(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Allows orchestrator instance to be called directly as a callable."""
        return self.process_query(user_query, **kwargs)