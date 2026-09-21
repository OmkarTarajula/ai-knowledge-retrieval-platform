

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.memory_agent import ConversationMemoryAgent


def run_tests():
    agent = ConversationMemoryAgent(window_size=2)  # Max 4 messages (2 turns)
    passed = 0
    total = 4

    print("=" * 65)
    print("RUNNING CONVERSATION MEMORY AGENT TEST SUITE")
    print("=" * 65)

    # 1. Test Initial Topic Extraction
    agent.add_interaction("user", "Tell me about MBA admission")
    agent.add_interaction("assistant", "MBA admission requires a bachelor's degree and entrance exam scores.")
    
    test_1 = agent.current_topic == "mba admission"
    if test_1:
        passed += 1
        print("[PASS] Test 01: Topic extraction correctly captured 'mba admission'")
    else:
        print(f"[FAIL] Test 01: Expected 'mba admission', got '{agent.current_topic}'")

    # 2. Test Follow-up Context Resolution
    resolved = agent.resolve_context("What are the eligibility requirements?")
    expected_resolution = "What are the eligibility requirements for mba admission?"
    test_2 = (resolved.lower() == expected_resolution.lower())
    if test_2:
        passed += 1
        print(f"[PASS] Test 02: Follow-up resolved -> '{resolved}'")
    else:
        print(f"[FAIL] Test 02: Expected '{expected_resolution}', got '{resolved}'")

    # 3. Test Context Switching (Clean transition, no topic bleed)
    agent.add_interaction("user", resolved)
    agent.add_interaction("assistant", "Minimum 50% marks in graduation are required.")
    
    # User switches topic explicitly to PC admission
    switch_query = "Now tell me about PC admission"
    resolved_switch = agent.resolve_context(switch_query)
    agent.add_interaction("user", switch_query)
    
    # Now ask follow-up for PC admission
    follow_switch = agent.resolve_context("What are the fees?")
    test_3 = "pc admission" in follow_switch.lower() and "mba" not in follow_switch.lower()
    if test_3:
        passed += 1
        print(f"[PASS] Test 03: Context switch handled cleanly -> '{follow_switch}'")
    else:
        print(f"[FAIL] Test 03: Bleed detected. Resolved to: '{follow_switch}'")

    # 4. Test Sliding-Window Capacity Constraint
    # We added 5 messages; with window_size=2, it must not exceed 4 items
    recent = agent.get_recent_context()
    test_4 = len(recent) <= 4
    if test_4:
        passed += 1
        print(f"[PASS] Test 04: Sliding window successfully capped at {len(recent)} items (max 4)")
    else:
        print(f"[FAIL] Test 04: History overflowed window limit. Count: {len(recent)}")

    print("=" * 65)
    print(f"RESULTS: {passed} Passed | {total - passed} Failed out of {total} tests")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()