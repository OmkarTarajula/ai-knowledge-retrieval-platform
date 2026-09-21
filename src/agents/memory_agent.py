import uuid
from typing import List, Dict, Any, Optional


class ConversationMemoryAgent:
    def __init__(self, window_size: int = 4):
        """
        :param window_size: Maximum number of recent message turns (user + assistant) to retain.
        """
        self.window_size = window_size
        self.conversation_id = str(uuid.uuid4())[:8]
        self.history: List[Dict[str, str]] = []
        self.current_topic: Optional[str] = None

    def add_interaction(self, role: str, content: Any) -> None:
        """
        Appends a message turn and maintains the sliding window size.
        Safely normalizes incoming text whether passed as str or dict.
        """
        if role not in ("user", "assistant"):
            raise ValueError("Role must be 'user' or 'assistant'.")

        # Safely convert dicts or other types into a string before stripping
        if isinstance(content, dict):
            content_str = str(content.get("answer", content.get("response", content.get("content", str(content)))))
        else:
            content_str = str(content)

        self.history.append({"role": role, "content": content_str.strip()})

        # Maintain sliding window to control prompt token load
        if len(self.history) > (self.window_size * 2):
            self.history = self.history[-(self.window_size * 2):]

        # Extract primary entity if coming from a direct user turn
        if role == "user":
            self._update_topic(content_str)

    def _update_topic(self, query: str) -> None:
        """
        Extracts active subject nouns or tags to handle topic shifts.
        """
        cleaned = query.strip().rstrip("?,.")
        lower_q = cleaned.lower()

        # Check for explicit topic markers
        triggers = ["about", "for", "regarding"]
        for t in triggers:
            if t in lower_q:
                parts = lower_q.split(t, 1)
                if len(parts) > 1 and len(parts[1].strip()) > 2:
                    self.current_topic = parts[1].strip()
                    return

        # Fallback to entire query if sufficiently specific
        words = lower_q.split()
        if len(words) >= 2 and not any(w in ("what", "how", "why", "when", "is", "are") for w in words):
            self.current_topic = cleaned

    def resolve_context(self, incoming_query: str) -> str:
        """
        Resolves follow-up queries that lack an explicit entity using prior turn context.
        If a new topic is detected, switches context cleanly without bleeding old state.
        """
        clean_query = incoming_query.strip()
        lower_query = clean_query.lower()
        words = lower_query.split()

        # Follow-up indicators that lack explicit subject context
        is_followup = (
            any(phrase in lower_query for phrase in [
                "what are the requirements",
                "what is the eligibility",
                "what are the fees",
                "how much does it cost",
                "what documents are required",
                "what is the syllabus",
                "tell me more"
            ])
            or any(pronoun in words for pronoun in ["it", "this", "that", "its"])
            or len(words) <= 5
        )

        # Context-switch check: If the query explicitly defines a new target, do not append old topic
        explicit_target_markers = ["admission", "course", "policy", "leave", "support"]
        has_new_target = any(m in lower_query for m in explicit_target_markers) and any(t in lower_query for t in ["tell me about", "what is", "explain"])

        if is_followup and not has_new_target and self.current_topic:
            # Re-anchor the follow-up question to the active topic
            stripped_query = clean_query.rstrip("?,.")
            return f"{stripped_query} for {self.current_topic}?"

        return clean_query

    def get_recent_context(self) -> List[Dict[str, str]]:
        """
        Returns recent conversational history for prompt augmentation.
        """
        return list(self.history)

    def clear(self) -> None:
        """
        Resets conversation state.
        """
        self.history = []
        self.current_topic = None
        self.conversation_id = str(uuid.uuid4())[:8]