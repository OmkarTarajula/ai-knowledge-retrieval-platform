

import sys
import os

# Ensure the root directory is accessible for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.clarification_agent import ClarificationAgent


def run_tests():
    agent = ClarificationAgent()
    passed = 0
    failed = 0

    # 15 Test cases matching mentor requirements
    test_cases = [
        # Ambiguous / Incomplete Queries (Should need clarification)
        {"id": 1, "query": "how do i apply", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Dangling action prefix"},
        {"id": 2, "query": "fees", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Single bare keyword"},
        {"id": 3, "query": "tell me about", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Dangling clause"},
        {"id": 4, "query": "leave policy", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Two-token broad topic"},
        {"id": 5, "query": "what is the process", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Incomplete procedural question"},
        {"id": 6, "query": "registration", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Broad entity without context"},
        {"id": 7, "query": "can i know", "intent": "AMBIGUOUS", "expect_clarify": True, "desc": "Dangling question stem"},

        # Multi-Part Queries (Clear enough to retrieve without clarification)
        {"id": 8, "query": "What is the eligibility for MBA and what are the fees?", "intent": "FACTUAL", "expect_clarify": False, "desc": "Multi-part query with 'and'"},
        {"id": 9, "query": "How do I apply for casual leave, and how many days are allowed?", "intent": "PROCEDURAL", "expect_clarify": False, "desc": "Multi-part procedural request"},
        {"id": 10, "query": "Tell me about course syllabus as well as placement records", "intent": "FACTUAL", "expect_clarify": False, "desc": "Multi-part with 'as well as'"},

        # Clear Single-Intent Queries (Should proceed directly to retrieval)
        {"id": 11, "query": "What is the annual casual leave allowance?", "intent": "FACTUAL", "expect_clarify": False, "desc": "Clear factual question"},
        {"id": 12, "query": "What are the steps to reset my enterprise portal password?", "intent": "PROCEDURAL", "expect_clarify": False, "desc": "Clear procedural question"},
        {"id": 13, "query": "What is the difference between sick leave and paid leave?", "intent": "COMPARATIVE", "expect_clarify": False, "desc": "Clear comparative question"},
        {"id": 14, "query": "What is the minimum attendance percentage required for exams?", "intent": "FACTUAL", "expect_clarify": False, "desc": "Specific factual policy query"},
        {"id": 15, "query": "Who is eligible for maternity leave benefits?", "intent": "FACTUAL", "expect_clarify": False, "desc": "Direct eligibility question"}
    ]

    print("=" * 65)
    print("RUNNING CLARIFICATION AGENT TEST SUITE (15 TEST PATTERNS)")
    print("=" * 65)

    for tc in test_cases:
        res = agent.evaluate(tc["query"], {"intent": tc["intent"]})
        is_ok = (res["needs_clarification"] == tc["expect_clarify"])

        if is_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"[{status}] Test {tc['id']:02d}: '{tc['query']}'")
        print(f"       Description: {tc['desc']}")
        print(f"       Needs Clarification: {res['needs_clarification']} (Expected: {tc['expect_clarify']})")
        if res["needs_clarification"]:
            print(f"       Follow-up Prompt: {res['follow_up_question']}")
        print("-" * 65)

    # Test Query Reformulation Logic
    print("\nTESTING QUERY REFORMULATION LOGIC:")
    reformulation_checks = [
        ("how do i apply", "casual leave", "How do I apply for casual leave?"),
        ("what are the fees", "MBA program", "What are the fees for MBA program?"),
        ("tell me about", "maternity policy", "Tell me about maternity policy.")
    ]

    for orig, user_input, expected in reformulation_checks:
        refined = agent.reformulate(orig, user_input)
        status = "PASS" if refined == expected else "FAIL"
        print(f"[{status}] Original: '{orig}' + Clarification: '{user_input}' -> '{refined}'")

    print("=" * 65)
    print(f"RESULTS: {passed} Passed | {failed} Failed out of {len(test_cases)} tests")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()