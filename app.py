import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tempfile
import streamlit as st

from src.ingestion.document_parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.vectorstore.store import VectorStoreManager
from src.pipeline.orchestrator import MultiAgentOrchestrator

# UI Page Layout
st.set_page_config(
    page_title="AI Knowledge Retrieval Platform",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session Backend Components
@st.cache_resource
def init_system():
    vstore = VectorStoreManager()
    parser = DocumentParser()
    chunker = TextChunker(chunk_size=400, chunk_overlap=40)
    orchestrator = MultiAgentOrchestrator(vector_store=vstore, default_top_k=3, threshold=0.45)
    return vstore, parser, chunker, orchestrator

vector_store, parser, chunker, orchestrator = init_system()

# Sidebar: Document Ingestion (Milestone 1)
with st.sidebar:
    st.header("📂 Document Ingestion")
    st.caption("Supports PDF, DOCX, TXT, CSV (Milestone 1)")

    domain_choice = st.selectbox(
        "Select Knowledge Domain:",
        ["HR & Operations", "Technical Support", "General"]
    )

    uploaded_files = st.file_uploader(
        "Upload Reference Documents",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    if st.button("Process & Index Documents"):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            total_chunks = 0
            with st.spinner("Parsing, chunking, and embedding documents..."):
                for uploaded_file in uploaded_files:
                    # Save temporary file for local parsing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        blocks, metadata = parser.parse_file(tmp_path, filename=uploaded_file.name, domain=domain_choice)
                        doc_chunks = chunker.chunk_document(blocks, metadata)
                        indexed_count = vector_store.add_chunks(doc_chunks)
                        total_chunks += indexed_count
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            st.success(f"Successfully processed {len(uploaded_files)} file(s) into {total_chunks} vector chunks!")

    st.divider()
    st.subheader("⚙️ Agent Settings (Milestone 2)")
    top_k_val = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=8, value=3)
    thresh_val = st.slider("Similarity Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)


# Main Panel: Multi-Agent Query Resolution System (Milestone 2)
st.title("🧠 AI-Based Knowledge Retrieval & Query Resolution")
st.write("An enterprise multi-agent retrieval system featuring grounded response generation, citations, and classification.")

user_query = st.text_input("Enter your query or prompt:", placeholder="e.g., How do I apply for casual leave?")

col1, col2 = st.columns([1, 4])
with col1:
    submit_btn = st.button("Resolve Query", type="primary")

if submit_btn and user_query.strip():
    with st.spinner("Orchestrating agents..."):
        result = orchestrator.process_query(
            user_query=user_query,
            domain=domain_choice,
            top_k=top_k_val,
            threshold=thresh_val
        )

    # 1. Agent Classification Banner
    analysis = result.get("query_analysis", {})
    q_type = analysis.get("query_type", "Unknown").upper()
    confidence = analysis.get("confidence", 0.0)

    st.markdown("### 🤖 Agent Pipeline Analysis")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Query Intent", q_type)
    col_b.metric("Classifier Confidence", f"{confidence * 100:.1f}%")
    col_c.metric("Pipeline Status", result.get("status", "completed").upper())

    # 2. Ambiguity Handling (Milestone 2 requirement)
    if result.get("status") == "clarification_needed":
        st.warning(f"⚠️ **Ambiguous Query Detected:** {result.get('answer')}")
    else:
        resp = result.get("response", {})
        
        # 3. Grounded Response
        st.markdown("### 💬 Grounded Response")
        st.write(resp.get("answer", "No answer generated."))

        # 4. Source Metadata & Citations (Milestone 2 requirement)
        st.markdown("### 📚 Source Citations & Attribution")
        sources = resp.get("sources", [])
        if sources:
            for s in sources:
                with st.expander(f"📄 {s['file_name']} (Page {s['page_number']} | {s['section']}) — Similarity: {s['similarity_score']}"):
                    st.write(f"**Chunk ID:** `{s['chunk_id']}`")
        else:
            st.info("No external document citations associated with this output.")