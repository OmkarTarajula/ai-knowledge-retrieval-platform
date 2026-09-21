import os
import urllib.parse
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ingestion, Vectorstore, and Pipeline modules
from src.ingestion.document_parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.vectorstore.store import VectorStoreManager
from src.pipeline.orchestrator import MultiAgentOrchestrator

# Milestone 3 Agents & Modules
from src.agents.clarification_agent import ClarificationAgent
from src.agents.memory_agent import ConversationMemoryAgent
from src.transparency_panel import ResponseTransparencyPanel
from src.voice_module import VoiceInteractionModule

# Page Configuration
st.set_page_config(page_title="AI Knowledge Retrieval Platform", layout="wide")

st.title("🤖 AI Multi-Agent Knowledge Retrieval Platform")
st.caption("Enterprise RAG system with interactive multi-turn, multi-modal assistance.")

# Session State Initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MultiAgentOrchestrator(vector_store=st.session_state.vector_store)

if "memory_agent" not in st.session_state:
    st.session_state.memory_agent = ConversationMemoryAgent(window_size=4)

if "clarification_agent" not in st.session_state:
    st.session_state.clarification_agent = ClarificationAgent()

if "transparency_panel" not in st.session_state:
    st.session_state.transparency_panel = ResponseTransparencyPanel()

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# Chat messages display list: retains user and assistant turns with chunk metadata for the UI feed
if "chat_display_history" not in st.session_state:
    st.session_state.chat_display_history = []


# --- SIDEBAR: Document Ingestion ---
with st.sidebar:
    st.header("📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    if st.button("Process & Index Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Extracting, chunking, and indexing vectors..."):
                all_chunks = []
                parser = DocumentParser()
                chunker = TextChunker()

                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(".", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    docs = parser.parse(temp_path)
                    chunks = chunker.chunk(docs)
                    all_chunks.extend(chunks)

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                st.session_state.vector_store.build_index(all_chunks)
                st.success(f"Successfully indexed {len(all_chunks)} chunks across {len(uploaded_files)} files!")

    if st.button("Clear Conversation Memory"):
        st.session_state.memory_agent.clear()
        st.session_state.chat_display_history = []
        st.session_state.pending_clarification = None
        st.info("Conversation history reset.")


# --- MAIN AREA: Voice Widget, Chat Stream, and Processing ---

# 1. Voice Speech-to-Text Widget
spoken_result = VoiceInteractionModule.render_speech_to_text_widget()

def execute_pipeline(query_text: str):
    """Runs query understanding, retrieval, response generation, updates memory and chat UI."""
    st.session_state.chat_display_history.append({"role": "user", "content": query_text})

    with st.spinner("Executing multi-agent resolution pipeline..."):
        if hasattr(st.session_state.orchestrator, "process_query"):
            result = st.session_state.orchestrator.process_query(query_text)
        elif hasattr(st.session_state.orchestrator, "run"):
            result = st.session_state.orchestrator.run(query_text)
        else:
            result = st.session_state.orchestrator(query_text)

        # Robust extraction: unwrap dictionaries so content is strictly a clean string
        if isinstance(result, dict):
            raw_ans = result.get("answer", result.get("response", ""))
            if isinstance(raw_ans, dict):
                response_text = str(raw_ans.get("answer", raw_ans.get("response", str(raw_ans))))
            else:
                response_text = str(raw_ans) if raw_ans else "No answer generated."

            retrieved_chunks = result.get("sources", result.get("retrieved_chunks", []))
        else:
            response_text = str(result)
            retrieved_chunks = []

        # Store clean string in conversational memory agent
        st.session_state.memory_agent.add_interaction("user", query_text)
        st.session_state.memory_agent.add_interaction("assistant", response_text)

        # Store clean string in chat display history
        st.session_state.chat_display_history.append({
            "role": "assistant",
            "content": response_text,
            "chunks": retrieved_chunks
        })


def handle_incoming_query(raw_query: str):
    """Handles context resolution and ambiguity evaluation before pipeline execution."""
    if not raw_query or not raw_query.strip():
        return

    contextual_query = st.session_state.memory_agent.resolve_context(raw_query.strip())
    clarification_check = st.session_state.clarification_agent.evaluate(contextual_query)

    if clarification_check["needs_clarification"]:
        st.session_state.pending_clarification = {
            "original_query": contextual_query,
            "question": clarification_check["follow_up_question"]
        }
    else:
        execute_pipeline(contextual_query)


# If audio was captured, offer an immediate 1-click execution button
if spoken_result:
    col_v1, col_v2 = st.columns([3, 1])
    with col_v1:
        edited_spoken = st.text_input("Review Spoken Query:", value=spoken_result, key="spoken_editable")
    with col_v2:
        st.write("")
        st.write("")
        if st.button("🚀 Run Spoken Query", type="primary"):
            handle_incoming_query(edited_spoken)
            st.rerun()


# 2. Render Multi-turn Conversational Chat Stream (Turn History Visibility)
st.markdown("---")
for msg in st.session_state.chat_display_history:
    with st.chat_message(msg["role"]):
        # Extract string if msg["content"] was previously stored as a dict
        raw_msg_content = msg["content"]
        if isinstance(raw_msg_content, dict):
            display_text = str(raw_msg_content.get("answer", raw_msg_content.get("response", str(raw_msg_content))))
        else:
            display_text = str(raw_msg_content)

        st.markdown(display_text)

        if msg["role"] == "assistant":
            # TTS audio playback controls
            VoiceInteractionModule.render_text_to_speech_controls(display_text)
            # Response Transparency Panel
            chunks = msg.get("chunks", [])
            if not chunks and isinstance(raw_msg_content, dict):
                chunks = raw_msg_content.get("sources", raw_msg_content.get("retrieved_chunks", []))
            st.session_state.transparency_panel.render(chunks)


# 3. Clarification Flow vs Chat Input Box
if st.session_state.pending_clarification:
    st.warning(f"⚠️ **Clarification Needed:** {st.session_state.pending_clarification['question']}")
    user_reply = st.text_input("Your Clarification / Details:", key="clarification_input_field")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Submit Clarification", type="primary"):
            if user_reply.strip():
                orig_q = st.session_state.pending_clarification["original_query"]
                refined_query = st.session_state.clarification_agent.reformulate(orig_q, user_reply)
                st.session_state.pending_clarification = None
                execute_pipeline(refined_query)
                st.rerun()
            else:
                st.error("Please enter a clarification response.")
    with col2:
        if st.button("Cancel Clarification"):
            st.session_state.pending_clarification = None
            st.rerun()
else:
    typed_query = st.chat_input("Ask a question about your documents...")
    if typed_query:
        handle_incoming_query(typed_query)
        st.rerun()