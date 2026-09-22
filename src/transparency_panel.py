

from typing import List, Dict, Any
import streamlit as st


class ResponseTransparencyPanel:
    def __init__(self, high_threshold: float = 0.70, medium_threshold: float = 0.45):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    @staticmethod
    def _extract_score(chunk: Dict[str, Any]) -> float:
        """Safely extracts similarity score across all agent and vector store schemas."""
        for key in ("similarity_score", "score", "similarity"):
            if key in chunk and chunk[key] is not None:
                try:
                    return float(chunk[key])
                except (ValueError, TypeError):
                    pass
        return 0.0

    def calculate_confidence(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Derives an aggregated confidence score and qualitative rating
        from the top retrieved chunks.
        """
        if not chunks:
            return {
                "level": "INSUFFICIENT EVIDENCE",
                "score": 0.0,
                "color": "#dc3545",  # Red
                "badge": "Insufficient Evidence"
            }

        scores = [self._extract_score(chunk) for chunk in chunks]
        top_score = max(scores) if scores else 0.0

        if top_score >= self.high_threshold:
            level = "HIGH"
            color = "#28a745"  # Green
        elif top_score >= self.medium_threshold:
            level = "MEDIUM"
            color = "#fd7e14"  # Orange
        else:
            level = "LOW / UNCERTAIN"
            color = "#dc3545"  # Red

        return {
            "level": level,
            "score": round(float(top_score), 4),
            "color": color,
            "badge": f"{level} ({round(top_score * 100, 1)}%)"
        }

    def render(self, retrieved_chunks: List[Dict[str, Any]]) -> None:
        """
        Draws the complete audit panel in Streamlit alongside the response.
        """
        confidence = self.calculate_confidence(retrieved_chunks)

        st.markdown("---")
        st.subheader("🔍 Response Transparency & Evidence Panel")

        # Confidence Indicator Badge
        st.markdown(
            f"**Retrieval Confidence:** "
            f"<span style='background-color:{confidence['color']}; color:white; "
            f"padding:3px 10px; border-radius:12px; font-weight:bold; font-size:0.9em;'>"
            f"{confidence['badge']}</span>",
            unsafe_allow_html=True
        )

        if confidence["level"] == "LOW / UNCERTAIN":
            st.warning(
                "Notice: The retrieval similarity score is low. "
                "The supporting evidence may be weak or incomplete."
            )

        if not retrieved_chunks:
            st.info("No supporting evidence documents were retrieved.")
            return

        st.markdown(f"**Retrieved Evidence Sources ({len(retrieved_chunks)} Chunks):**")

        # Display chunks inside collapsible expanders
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})
            source_doc = (
                metadata.get("file_name")
                or metadata.get("source")
                or chunk.get("file_name")
                or "Unknown Source"
            )
            page_num = metadata.get("page_number", metadata.get("page"))
            section = metadata.get("section", metadata.get("row"))
            if page_num and section:
                page_or_sec = f"{section} (Page {page_num})" if str(page_num) not in str(section) else str(section)
            else:
                page_or_sec = str(section or (f"Page {page_num}" if page_num else "N/A"))

            chunk_id = (
                chunk.get("chunk_id")
                or metadata.get("chunk_id")
                or f"chk-{idx:02d}"
            )
            score = self._extract_score(chunk)
            text_snippet = chunk.get("text", chunk.get("content", "No content available.")).strip()

            expander_title = (
                f"Source [{idx}]: {source_doc} "
                f"| Location: {page_or_sec} "
                f"| Similarity: {score:.4f}"
            )

            with st.expander(expander_title, expanded=False):
                st.markdown(f"**Chunk Identifier:** `{chunk_id}`")
                st.markdown(f"**Source Document:** `{source_doc}`")
                st.markdown(f"**Section / Page / Row:** `{page_or_sec}`")
                st.markdown(f"**Cosine Similarity Score:** `{score:.4f}`")
                st.markdown("**Evidence Text Content:**")
                st.info(text_snippet)