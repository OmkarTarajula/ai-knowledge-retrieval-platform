import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.query_understanding_agent import QueryUnderstandingAgent

def run_query_classification_evaluation():
    agent = QueryUnderstandingAgent()

    # 20 benchmark test queries across HR/Operations & Tech Support domains
    test_cases = [
        # 10 Factual Queries
        {"query": "What is the standard probation period for new employees?", "expected": "factual"},
        {"query": "What are the official working hours for technical support?", "expected": "factual"},
        {"query": "Who is eligible for annual health insurance coverage?", "expected": "factual"},
        {"query": "What is the maximum limit for medical expense reimbursement?", "expected": "factual"},
        {"query": "What are the core database credentials required for staging?", "expected": "factual"},
        {"query": "What is the emergency contact number for IT helpdesk?", "expected": "factual"},
        {"query": "How many days of casual leave do employees receive per year?", "expected": "factual"},
        {"query": "Where is the company asset policy documented?", "expected": "factual"},
        {"query": "What is the notice period required during resignation?", "expected": "factual"},
        {"query": "What are the primary responsibilities of an on-call engineer?", "expected": "factual"},

        # 5 Procedural Queries
        {"query": "How do I apply for annual earned leave in the portal?", "expected": "procedural"},
        {"query": "What are the steps to submit travel expense claims?", "expected": "procedural"},
        {"query": "How can I reset my corporate VPN password?", "expected": "procedural"},
        {"query": "How to deploy the application build to production servers?", "expected": "procedural"},
        {"query": "What is the procedure to request a new development laptop?", "expected": "procedural"},

        # 3 Comparative Queries
        {"query": "What is the difference between casual leave and sick leave?", "expected": "comparative"},
        {"query": "Compare standard health insurance vs premium family coverage.", "expected": "comparative"},
        {"query": "What are the advantages of permanent remote work versus hybrid model?", "expected": "comparative"},

        # 2 Ambiguous Queries (Flagged for clarification)
        {"query": "tell me about leave", "expected": "ambiguous"},
        {"query": "policy", "expected": "ambiguous"}
    ]

    print("==================================================")
    print("   RUNNING QUERY UNDERSTANDING EVALUATION SUITE   ")
    print("==================================================\n")

    correct = 0
    total = len(test_cases)

    for idx, test in enumerate(test_cases, 1):
        result = agent.analyze_query(test["query"])
        predicted = result.get("query_type")
        is_match = predicted == test["expected"]

        if is_match:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] Test {idx:02d}: \"{test['query']}\"")
        print(f"       Expected: {test['expected']} | Predicted: {predicted} (Confidence: {result['confidence'] * 100:.0f}%)\n")

    accuracy = (correct / total) * 100
    print("--------------------------------------------------")
    print(f"Evaluation Complete: {correct}/{total} Passed | Accuracy: {accuracy:.2f}%")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_query_classification_evaluation()