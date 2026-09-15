import re
from typing import Dict, Any

class QueryUnderstandingAgent:
    """
    Classifies incoming user queries into one of four categories:
    Factual, Procedural, Comparative, or Ambiguous.
    Produces structured output with classification confidence and routing metadata.
    """

    def __init__(self):
        # Rule patterns matching mentor's requirements
        self.procedural_keywords = [
            r"\bhow to\b", r"\bhow do i\b", r"\bhow can i\b", r"\bsteps to\b",
            r"\bprocedure\b", r"\bprocess for\b", r"\bapply for\b", r"\binstructions\b"
        ]
        self.comparative_keywords = [
            r"\bdifference between\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
            r"\bsimilarities\b", r"\badvantages of\b", r"\bhow is .* different\b"
        ]
        self.factual_keywords = [
            r"\bwhat is\b", r"\bwhat are\b", r"\bwhen was\b", r"\bwho is\b",
            r"\bwhere is\b", r"\bhow many\b", r"\beligible\b", r"\bdefinition\b", r"\bpolicy\b"
        ]

    def analyze_query(self, query: str) -> Dict[str, Any]:
        cleaned_query = query.strip()
        query_lower = cleaned_query.lower()
        words = query_lower.split()

        # Check for ambiguity (underspecified or vague input)
        if len(words) <= 3 or query_lower in ["tell me about leave", "leave policy", "policy", "help", "leave", "info"]:
            return {
                "query": cleaned_query,
                "query_type": "ambiguous",
                "confidence": 0.88,
                "routing": "clarification",
                "explanation": "Query is underspecified or lacks distinct intent. Flagged for clarification."
            }

        # Check for comparative queries
        for pattern in self.comparative_keywords:
            if re.search(pattern, query_lower):
                return {
                    "query": cleaned_query,
                    "query_type": "comparative",
                    "confidence": 0.94,
                    "routing": "retrieval",
                    "explanation": "User is contrasting two or more entities or concepts."
                }

        # Check for procedural queries
        for pattern in self.procedural_keywords:
            if re.search(pattern, query_lower):
                return {
                    "query": cleaned_query,
                    "query_type": "procedural",
                    "confidence": 0.92,
                    "routing": "retrieval",
                    "explanation": "User is seeking sequential step-by-step instructions."
                }

        # Check for factual queries
        for pattern in self.factual_keywords:
            if re.search(pattern, query_lower):
                return {
                    "query": cleaned_query,
                    "query_type": "factual",
                    "confidence": 0.90,
                    "routing": "retrieval",
                    "explanation": "User is requesting a specific factual data point or definition."
                }

        # Default fallback
        return {
            "query": cleaned_query,
            "query_type": "factual",
            "confidence": 0.75,
            "routing": "retrieval",
            "explanation": "Defaulted to general factual retrieval."
        }