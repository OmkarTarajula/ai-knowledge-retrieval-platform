# AI Knowledge Retrieval Platform

An end-to-end Retrieval-Augmented Generation (RAG) platform designed to parse unstructured documents, generate semantic embeddings, and resolve user queries using context-aware LLM orchestration.

---

## Key Features
* Document Ingestion: Automated document parsing and recursive text chunking.
* Vector Search Engine: High-speed vector similarity matching for document retrieval.
* Query Resolution: Intelligent context injection into prompt templates to prevent hallucinations.

---

## Architecture Flow

User Query ---> Orchestrator
                     |
                     v
Raw Documents ---> Parser ---> Chunker ---> Vector Store
                                               |
                                               v (Retrieve Top Chunks)
                                       Context + Prompt
                                               |
                                               v
                                        Final AI Answer

---

## Project Structure
* src/ingestion/ - Document parsing and chunking scripts.
* src/vectorstore/ - Embedding models and vector index storage.
* src/orchestration/ - Pipeline logic and prompt orchestration.
* app.py - Main execution file.
* requirements.txt - Project dependencies.

---

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Start the application: python app.py