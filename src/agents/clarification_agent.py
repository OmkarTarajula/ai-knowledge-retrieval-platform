from typing import Dict, Any, Optional


class ClarificationAgent:
    def __init__(self):
        # Fallback question templates mapped to specific ambiguity causes
        self.clarification_prompts = {
            "dangling_action": "Could you specify what you want to apply for or get details on?",
            "under_specified": "Could you provide a bit more detail about what specifically you need?",
            "entity_ambiguity": "Are you asking about policy rules, application steps, or fee structure?",
            "multipart_ambiguous": "Your request has multiple parts. Which specific aspect should we prioritize first?"
        }

    def evaluate(self, query: str, intent_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates an incoming query to determine if vector retrieval should proceed
        or if an interactive clarification question is needed first.
        """
        raw_query = (query or "").strip()
        tokens = [t.lower() for t in raw_query.split() if t]
        token_count = len(tokens)

        # 1. Intent check from QueryUnderstandingAgent or explicit payload
        intent = "UNKNOWN"
        if intent_payload and isinstance(intent_payload, dict):
            raw_intent = intent_payload.get("intent") or intent_payload.get("query_type") or ""
            intent = str(raw_intent).upper()
            if intent_payload.get("routing") == "clarification":
                intent = "AMBIGUOUS"

        is_flagged_ambiguous = (intent == "AMBIGUOUS")

        # 2. Check for multi-part conjunctions
        conjunctions = {"and", "also", "plus", "as well as", "&"}
        has_multiple_intents = any(c in tokens for c in conjunctions) or raw_query.count("?") > 1

        # 3. Structural checks: incomplete dangling clauses or excessively short tokens
        incomplete_prefixes = (
            "how do i apply",
            "tell me about",
            "what is the process",
            "how to get",
            "can i know"
        )
        lower_query = raw_query.lower()
        is_dangling = any(lower_query.startswith(p) for p in incomplete_prefixes) and token_count <= 4
        is_bare_phrase = token_count <= 2

        # Make resolution decision
        needs_clarification = is_flagged_ambiguous or is_dangling or is_bare_phrase

        # Pick appropriate follow-up prompt
        if is_dangling:
            issue_type = "dangling_action"
            follow_up = self.clarification_prompts["dangling_action"]
        elif is_bare_phrase:
            issue_type = "under_specified"
            follow_up = f"Could you specify what details you need regarding '{raw_query}'?"
        elif is_flagged_ambiguous:
            issue_type = "entity_ambiguity"
            follow_up = self.clarification_prompts["entity_ambiguity"]
        elif has_multiple_intents:
            issue_type = "multi_part_clear"
            follow_up = None
        else:
            issue_type = "none"
            follow_up = None

        return {
            "needs_clarification": needs_clarification,
            "is_multipart": has_multiple_intents,
            "issue_type": issue_type,
            "follow_up_question": follow_up
        }

    def reformulate(self, initial_query: str, user_clarification: str) -> str:
        """
        Merges the user's clarification with the original query into a unified,
        complete query string ready for vector search.
        """
        clean_init = initial_query.strip().rstrip("?., ")
        clean_clarify = user_clarification.strip().rstrip("?., ")

        lower_init = clean_init.lower()

        # Contextual reconstructions
        if lower_init.startswith("how do i apply"):
            return f"How do I apply for {clean_clarify}?"

        if lower_init.startswith("tell me about"):
            return f"Tell me about {clean_clarify}."

        if any(lower_init.startswith(p) for p in ("what are the fees", "what is the cost", "cost of")):
            return f"What are the fees for {clean_clarify}?"

        if lower_init.startswith("what is the process of") or lower_init.startswith("what is the process"):
            return f"What is the process of {clean_clarify}?"

        if lower_init.startswith("can i know"):
            return f"Can I know about {clean_clarify}?"

        # General connector
        return f"{clean_init} regarding {clean_clarify}"