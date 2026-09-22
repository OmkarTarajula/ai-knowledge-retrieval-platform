"""
tests/test_rag_pipeline.py
End-to-End RAG Pipeline Integration Test Suite.
Validates ingestion, chunking, vector indexing, multi-agent orchestration,
clarification, conversational memory, and transparency reporting.
"""

import os
import sys
import shutil
import unittest

# Ensure the root directory is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.document_parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.vectorstore.store import VectorStoreManager
from src.pipeline.orchestrator import MultiAgentOrchestrator
from src.agents.clarification_agent import ClarificationAgent
from src.agents.memory_agent import ConversationMemoryAgent
from src.transparency_panel import ResponseTransparencyPanel


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_test_db"))
        os.makedirs(cls.test_db_dir, exist_ok=True)
        cls.hr_policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hr_policy.txt"))
        cls.tech_support_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tech_support.csv"))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_dir):
            shutil.rmtree(cls.test_db_dir, ignore_errors=True)

    def test_01_document_parsing_and_chunking(self):
        """Test document parsing returns (blocks, metadata) tuple and chunker produces valid chunks."""
        # 1. Parse TXT
        blocks, meta = DocumentParser.parse_file(self.hr_policy_path, filename="hr_policy.txt", domain="HR")
        self.assertIsInstance(blocks, list)
        self.assertIsInstance(meta, dict)
        self.assertGreater(len(blocks), 0)
        self.assertEqual(meta["file_name"], "hr_policy.txt")
        self.assertEqual(meta["domain"], "HR")

        # 2. Chunk TXT
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(blocks, meta)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

        first_chunk = chunks[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("metadata", first_chunk)
        self.assertEqual(first_chunk["metadata"]["file_name"], "hr_policy.txt")
        self.assertEqual(first_chunk["metadata"]["domain"], "HR")

        # 3. Parse CSV
        csv_blocks, csv_meta = DocumentParser.parse_file(self.tech_support_path, filename="tech_support.csv", domain="IT")
        self.assertGreater(len(csv_blocks), 0)
        csv_chunks = chunker.chunk_document(csv_blocks, csv_meta)
        self.assertGreater(len(csv_chunks), 0)

    def test_02_vectorstore_build_index_and_query(self):
        """Test vector store build_index, add_chunks, and query with similarity scores."""
        vs = VectorStoreManager(persist_directory=self.test_db_dir)

        # Ingest test chunks
        blocks, meta = DocumentParser.parse_file(self.hr_policy_path, filename="hr_policy.txt", domain="General")
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(blocks, meta)

        indexed_count = vs.build_index(chunks, clear_existing=True)
        self.assertEqual(indexed_count, len(chunks))

        # Query vector store
        query_results = vs.query("annual casual leave allowance", top_k=2)
        self.assertGreater(len(query_results), 0)
        top_res = query_results[0]
        self.assertIn("chunk_id", top_res)
        self.assertIn("text", top_res)
        self.assertIn("metadata", top_res)
        self.assertIn("similarity_score", top_res)
        self.assertGreater(top_res["similarity_score"], 0.0)

    def test_03_orchestrator_end_to_end_factual_query(self):
        """Test end-to-end factual resolution with complete return schema."""
        vs = VectorStoreManager(persist_directory=self.test_db_dir)
        blocks, meta = DocumentParser.parse_file(self.hr_policy_path, filename="hr_policy.txt")
        chunker = TextChunker()
        chunks = chunker.chunk_document(blocks, meta)
        vs.build_index(chunks, clear_existing=True)

        orchestrator = MultiAgentOrchestrator(vector_store=vs)
        result = orchestrator.process_query("What is the annual casual leave allowance?")

        # Check all required top-level attributes
        self.assertEqual(result["status"], "success")
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)
        self.assertIn("sources", result)
        self.assertIn("retrieved_chunks", result)
        self.assertGreater(len(result["retrieved_chunks"]), 0)
        self.assertIn("confidence_score", result)
        self.assertIn("confidence_level", result)
        self.assertIn(result["confidence_level"], ["High", "Medium", "Low"])

        # Test callable and run aliases
        run_res = orchestrator.run("What is the annual casual leave allowance?")
        self.assertEqual(run_res["status"], "success")
        call_res = orchestrator("What is the annual casual leave allowance?")
        self.assertEqual(call_res["status"], "success")

    def test_04_orchestrator_comparative_and_procedural_queries(self):
        """Test response generation adapts properly to procedural and comparative queries."""
        vs = VectorStoreManager(persist_directory=self.test_db_dir)
        blocks, meta = DocumentParser.parse_file(self.hr_policy_path, filename="hr_policy.txt")
        vs.build_index(TextChunker().chunk_document(blocks, meta), clear_existing=True)

        orchestrator = MultiAgentOrchestrator(vector_store=vs)

        # Procedural query
        proc_res = orchestrator.process_query("How do I apply for casual leave?")
        self.assertEqual(proc_res["status"], "success")
        self.assertEqual(proc_res["query_analysis"]["query_type"], "procedural")
        self.assertIn("steps", proc_res["answer"].lower())

        # Comparative query
        comp_res = orchestrator.process_query("What is the difference between casual leave and sick leave?")
        self.assertEqual(comp_res["status"], "success")
        self.assertEqual(comp_res["query_analysis"]["query_type"], "comparative")
        self.assertIn("comparison", comp_res["answer"].lower())

    def test_05_clarification_agent_integration(self):
        """Test clarification detection when query is ambiguous and query reformulate."""
        clarification_agent = ClarificationAgent()

        # Underspecified ambiguous query
        eval_result = clarification_agent.evaluate("leave policy")
        self.assertTrue(eval_result["needs_clarification"])
        self.assertIsNotNone(eval_result["follow_up_question"])

        # Integrated with QueryUnderstandingAgent payload
        orch = MultiAgentOrchestrator(vector_store=VectorStoreManager(self.test_db_dir))
        analysis = orch.query_agent.analyze_query("policy")
        eval_with_payload = clarification_agent.evaluate("policy", analysis)
        self.assertTrue(eval_with_payload["needs_clarification"])

        # Orchestrator routing to clarification
        clarify_res = orch.process_query("policy")
        self.assertEqual(clarify_res["status"], "clarification_needed")
        self.assertIn("answer", clarify_res)

        # Test query reformulation
        refined = clarification_agent.reformulate("how do i apply", "casual leave")
        self.assertEqual(refined, "How do I apply for casual leave?")

    def test_06_memory_agent_multi_turn_resolution(self):
        """Test conversation memory agent turn tracking, topic resolution, and sliding window."""
        memory = ConversationMemoryAgent(window_size=2)

        # Initial turn
        memory.add_interaction("user", "Tell me about MBA admission")
        memory.add_interaction("assistant", "MBA admission requires a bachelor's degree.")
        self.assertEqual(memory.current_topic, "mba admission")

        # Follow-up context resolution
        resolved = memory.resolve_context("What are the eligibility requirements?")
        self.assertIn("mba admission", resolved.lower())

        # Context switch
        memory.add_interaction("user", resolved)
        memory.add_interaction("assistant", "50% marks are required.")
        switch_query = "Now tell me about PC admission"
        memory.resolve_context(switch_query)
        memory.add_interaction("user", switch_query)
        memory.add_interaction("assistant", "Details for PC admission.")

        follow_pc = memory.resolve_context("What are the fees?")
        self.assertIn("pc admission", follow_pc.lower())
        self.assertNotIn("mba", follow_pc.lower())

        # Sliding window constraint (2 turns = max 4 items)
        self.assertLessEqual(len(memory.get_recent_context()), 4)

    def test_07_transparency_panel_confidence_calculation(self):
        """Test transparency panel correctly extracts similarity_score and produces accurate ratings."""
        panel = ResponseTransparencyPanel(high_threshold=0.70, medium_threshold=0.45)

        # High confidence test
        sample_chunks_high = [
            {
                "chunk_id": "doc1_chk1",
                "text": "Full leave policy text",
                "similarity_score": 0.82,
                "metadata": {"file_name": "hr.txt", "page_number": 1, "section": "Leave"}
            }
        ]
        conf_high = panel.calculate_confidence(sample_chunks_high)
        self.assertEqual(conf_high["level"], "HIGH")
        self.assertEqual(conf_high["score"], 0.82)
        self.assertEqual(conf_high["color"], "#28a745")

        # Medium confidence test
        sample_chunks_med = [
            {
                "chunk_id": "doc1_chk2",
                "text": "Medical reimbursement info",
                "similarity_score": 0.58,
                "metadata": {"file_name": "hr.txt", "page_number": 2, "section": "Benefits"}
            }
        ]
        conf_med = panel.calculate_confidence(sample_chunks_med)
        self.assertEqual(conf_med["level"], "MEDIUM")
        self.assertEqual(conf_med["score"], 0.58)

        # Empty chunks test
        conf_empty = panel.calculate_confidence([])
        self.assertEqual(conf_empty["level"], "INSUFFICIENT EVIDENCE")
        self.assertEqual(conf_empty["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
