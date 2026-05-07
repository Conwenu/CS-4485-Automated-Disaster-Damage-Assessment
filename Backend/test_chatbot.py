"""
Comprehensive test suite for Disaster Damage Assessment Chatbot.
Run with: python test_chatbot.py
"""

import json
from typing import Dict, Any, List, Tuple

# Import your services (adjust import paths as needed)
from app.services.chat_service import ChatService
from app.data.data import retrieve_true_data


class ChatbotTester:
    def __init__(self):
        self.chat_service = ChatService()
        self.session_id = "test-session-001"
        self.conversation_history: List[Dict[str, str]] = []
        self.passed = 0
        self.failed = 0
        self.total = 0

    def run_test(self, query: str, expected_intent: str, description: str, 
                 check_fn=None) -> Dict[str, Any]:
        """Run a single test query and validate the response."""
        self.total += 1
        print(f"\n{'='*60}")
        print(f"TEST {self.total}: {description}")
        print(f"Query: '{query}'")
        print(f"Expected Intent: {expected_intent}")
        print("-" * 40)

        # Send query
        response = self.chat_service.process_query(
            query=query,
            session_id=self.session_id,
            history=self.conversation_history
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({
            "role": "assistant",
            "content": response["response"]["text"]
        })

        # Extract actual intent
        actual_intent = response.get("parsed", {}).get("intent", "UNKNOWN")
        response_text = response["response"]["text"]
        requires_clarification = response.get("requires_clarification", False)

        # Display results
        print(f"Actual Intent:   {actual_intent}")
        print(f"Clarification:   {requires_clarification}")
        print(f"Response: {response_text[:200]}..." if len(response_text) > 200 else f"Response: {response_text}")

        # Validate
        passed = actual_intent == expected_intent
        if check_fn:
            passed = passed and check_fn(response)

        if passed:
            self.passed += 1
            print("✅ PASSED")
        else:
            self.failed += 1
            print("❌ FAILED")

        return response

    def run_clarification_followup(self, query: str, expected_intent: str, 
                                   description: str) -> Dict[str, Any]:
        """Run a follow-up query that provides missing parameters."""
        # The previous response should have pending_clarification
        # In a real frontend flow, you'd send that back. Here we simulate by
        # just sending the answer; the ChatService will handle it if the previous
        # response had pending_clarification (but our test doesn't store it).
        # For simplicity, we'll just test the follow-up as a new query.
        return self.run_test(query, expected_intent, description)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{self.total} passed, {self.failed} failed")
        print("="*60)


def main():
    tester = ChatbotTester()

    # ==================== BASIC QUERIES ====================
    
    # 1. City damage summary
    tester.run_test(
        query="What damage was detected in Santa Rosa?",
        expected_intent="GET_DAMAGE_FOR_LOCATION",
        description="City-level damage summary"
    )

    # 2. Overall damage distribution (no city)
    tester.run_test(
        query="What is the overall damage distribution?",
        expected_intent="GET_DAMAGE_DISTRIBUTION",
        description="Overall damage distribution (city=null)"
    )

    # 3. City-specific damage distribution
    tester.run_test(
        query="Show damage distribution for Miami",
        expected_intent="GET_DAMAGE_DISTRIBUTION",
        description="City-specific damage distribution"
    )

    # 4. Dataset health (failed processing)
    tester.run_test(
        query="How many records have failed processing?",
        expected_intent="GET_DATASET_HEALTH",
        description="Dataset health - failed records"
    )

    # ==================== MODEL PERFORMANCE ====================

    # 5. Model performance for a city
    tester.run_test(
        query="How accurate is the model in Houston?",
        expected_intent="GET_MODEL_PERFORMANCE",
        description="Model accuracy for a city"
    )

    # 6. Misclassifications
    tester.run_test(
        query="Where did the model make mistakes?",
        expected_intent="GET_MISCLASSIFICATIONS",
        description="Misclassifications (full dataset)"
    )

    # 7. Misclassifications for a city
    tester.run_test(
        query="Show me failure cases in Santa Rosa",
        expected_intent="GET_FAILURE_CASES",
        description="Failure cases for a specific city"
    )

    # 8. Accuracy by damage level
    tester.run_test(
        query="How accurate is the model for destroyed buildings?",
        expected_intent="GET_ACCURACY_BY_DAMAGE",
        description="Accuracy for a specific damage level"
    )

    # ==================== CONFIDENCE ANALYSIS ====================

    # 9. Confidence analysis for a city
    tester.run_test(
        query="What is the average confidence in Miami?",
        expected_intent="GET_CONFIDENCE_ANALYSIS",
        description="Average confidence for a city"
    )

    # 10. Confidence outliers (below threshold)
    tester.run_test(
        query="Which predictions have confidence below 0.6?",
        expected_intent="GET_CONFIDENCE_OUTLIERS",
        description="Low confidence outliers"
    )

    # 11. Highest confidence building
    tester.run_test(
        query="Which building is the model most sure about?",
        expected_intent="GET_HIGHEST_CONFIDENCE",
        description="Highest confidence building"
    )

    # ==================== TOP-K & FILTERING ====================

    # 12. Top K damaged buildings
    tester.run_test(
        query="Show the top 5 most damaged buildings in Santa Rosa",
        expected_intent="GET_TOP_K_DAMAGE",
        description="Top K damaged buildings"
    )

    # 13. Filter by damage level
    tester.run_test(
        query="Show me all destroyed buildings in Santa Rosa",
        expected_intent="GET_BUILDINGS_BY_DAMAGE",
        description="Filter buildings by damage level"
    )

    # 14. Filter by status (failed)
    tester.run_test(
        query="Show me buildings that failed to process in Santa Rosa",
        expected_intent="FILTER_BY_STATUS",
        description="Filter buildings by processing status"
    )

    # ==================== COMPARISONS & RANKING ====================

    # 15. Compare cities
    tester.run_test(
        query="Compare Houston and Miami",
        expected_intent="COMPARE_LOCATIONS",
        description="Compare two cities"
    )

    # 16. Compare buildings
    tester.run_test(
        query="Compare building santa-rosa-00000002_bldg0 and santa-rosa-00000002_bldg5",
        expected_intent="COMPARE_BUILDINGS",
        description="Compare two specific buildings"
    )

    # 17. Rank cities by damage
    tester.run_test(
        query="Which city was hit hardest?",
        expected_intent="RANK_CITIES_BY_DAMAGE",
        description="Rank cities by destruction"
    )

    # ==================== SCENE & BUILDING DETAILS ====================

    # 18. Scene summary
    tester.run_test(
        query="What's the damage in scene santa-rosa-00000002?",
        expected_intent="GET_SCENE_SUMMARY",
        description="Scene-level summary"
    )

    # 19. Building details
    tester.run_test(
        query="Tell me about building santa-rosa-00000002_bldg0",
        expected_intent="GET_BUILDING_DETAILS",
        description="Detailed building information"
    )

    # 20. Model explanation (reasoning)
    tester.run_test(
        query="Explain why building santa-rosa-00000002_bldg0 was destroyed",
        expected_intent="GET_MODEL_EXPLANATION",
        description="Model reasoning for a building"
    )

    # 21. Random building
    tester.run_test(
        query="Show me a random building in Miami",
        expected_intent="GET_RANDOM_BUILDING",
        description="Random building exploration"
    )

    # ==================== OFF-TOPIC & CLARIFICATION ====================

    # 22. Off-topic query
    tester.run_test(
        query="What is the weather today?",
        expected_intent="OUT_OF_SCOPE",
        description="Off-topic rejection"
    )

    # 23. Clarification needed (incomplete query)
    response = tester.run_test(
        query="Show me the top damaged buildings",
        expected_intent="GET_TOP_K_DAMAGE",  # Parser should identify intent but trigger clarification
        description="Clarification: missing top_k parameter"
    )

    # Check if clarification was requested
    if response.get("requires_clarification"):
        print("\n--> Clarification requested as expected.")
    else:
        print("\n--> WARNING: Clarification not requested for incomplete query.")

    # ==================== EXTERNAL KNOWLEDGE ====================
    
    # 24. Query requiring external knowledge (if service is implemented)
    tester.run_test(
        query="What is major damage?",
        expected_intent="OUT_OF_SCOPE",  # Falls back because no dataset operation
        description="External knowledge: definition query"
    )

    # 25. Mixed query: dataset + external knowledge
    tester.run_test(
        query="What is major damage in Santa Rosa?",
        expected_intent="GET_DAMAGE_FOR_LOCATION",
        description="Mixed: damage summary + external knowledge"
    )

    # ==================== SUMMARY ====================
    tester.print_summary()


if __name__ == "__main__":
    main()