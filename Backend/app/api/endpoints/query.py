from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from colorama import Fore, init
import time

from app.data.data import (
    retrieve_test_queries,
    get_new_damage_records,
    retrieve_true_data,
)
from app.services.query_parser import QueryParser
from app.services.query_engine import QueryEngine
from app.services.intent_dispatcher import IntentDispatcher
from app.services.response_service import ResponseService
from app.services.follow_up_detector import FollowUpDetector
from app.services.query_rewriter import QueryRewriter
from app.services.chat_service import ChatService
from app.models.models import ChatMessage, TestResultResponse, QueryRequest

init(autoreset=True)

router = APIRouter()

qp = QueryParser()
dataset = get_new_damage_records()
true_dataset = retrieve_true_data()
qe = QueryEngine(true_dataset)
dispatcher = IntentDispatcher(qe)
response_service = ResponseService()
follow_up_detector = FollowUpDetector()
rewriter = QueryRewriter()
chat_service = ChatService()


def format_history(history: List[ChatMessage]) -> str:
    """Convert a list of ChatMessage objects into a plain string."""
    if not history:
        return ""
    return "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in history])


def evaluate_parser(parser):
    """Evaluate QueryParser against predefined test cases."""
    TEST_CASES = retrieve_test_queries()
    total = len(TEST_CASES)
    correct_intent = 0
    correct_params = 0
    results = []
    start_time = time.time()

    for i, case in enumerate(TEST_CASES):
        result = parser.parse(case["query"])
        intent = result.get("intent")
        params = {k: v for k, v in result.items() if k != "intent"}

        expected_intent = case["expected_intent"]
        expected_fields = case["expected_fields"]

        intent_match = intent == expected_intent
        param_match = all(
            params.get(key) == expected_fields[key] for key in expected_fields
        )

        results.append(
            {
                "test_case": i + 1,
                "query": case["query"],
                "expected_intent": expected_intent,
                "predicted_intent": intent,
                "expected_parameters": expected_fields,
                "predicted_parameters": params,
                "intent_correct": intent_match,
                "parameters_correct": param_match,
            }
        )

        # Console output (optional – useful for debugging)
        print(Fore.YELLOW + f"Test Case {i + 1}")
        print(Fore.CYAN + f"Query: {case['query']}")
        print(Fore.GREEN + f"Expected Intent: {expected_intent}")
        print(Fore.GREEN + f"Predicted Intent: {intent}")
        print(Fore.GREEN + f"Expected Fields: {expected_fields}")
        print(Fore.GREEN + f"Predicted Fields: {params}")
        print(
            Fore.GREEN + "Intent Correct: Yes"
            if intent_match
            else Fore.RED + "Intent Correct: No"
        )
        print(
            Fore.GREEN + "Parameters Correct: Yes"
            if param_match
            else Fore.RED + "Parameters Correct: No"
        )
        print(Fore.WHITE + "-" * 50)

        if intent_match:
            correct_intent += 1
        if param_match:
            correct_params += 1

    duration = time.time() - start_time
    intent_acc = correct_intent / total
    param_acc = correct_params / total

    print(Fore.YELLOW + "\n=========================")
    print(Fore.GREEN + f"Intent Accuracy: {correct_intent}/{total} = {intent_acc:.2%}")
    print(
        Fore.GREEN + f"Parameter Accuracy: {correct_params}/{total} = {param_acc:.2%}"
    )
    print(Fore.YELLOW + "=========================")

    return {
        "intent_accuracy": f"{correct_intent}/{total} = {intent_acc:.2%}",
        "parameter_accuracy": f"{correct_params}/{total} = {param_acc:.2%}",
        "total_time_taken": f"{duration:.4f} seconds",
        "results": results,
    }


@router.post("/parseQuery/")
async def parseQuery(req: str):
    """Simple endpoint to test the parser alone."""
    print(f"Req: {req}")
    return qp.parse(req)


@router.post("/dispatchQuery/")
async def dispatchQuery(req: str):
    """Parse and dispatch a query without conversation handling."""
    print(f"Query To Be Dispatched: {req}")
    parsed = qp.parse(req)
    response = dispatcher.dispatch(parsed)
    return response


@router.post("/ask")
def ask(request: QueryRequest):
    try:
        return chat_service.process_query(
            query=request.query,
            session_id=request.session_id,
            history=(
                [m.model_dump() for m in request.history] if request.history else None
            ),
            pending_clarification=request.pending_clarification,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """Streaming version of /ask. Returns SSE stream."""
    try:
        return StreamingResponse(
            chat_service.process_query_stream(
                query=request.query,
                session_id=request.session_id,
                history=(
                    [m.model_dump() for m in request.history]
                    if request.history
                    else None
                ),
                pending_clarification=request.pending_clarification,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parser-test")
def parser_test():
    """Test only the QueryParser against predefined queries."""
    parser = QueryParser()
    return evaluate_parser(parser)


@router.get("/test-results", response_model=TestResultResponse)
def test_results():
    """Alias for /parser-test returning structured response."""
    parser = QueryParser()
    return evaluate_parser(parser)


@router.get("/showTest")
def showTest():
    """Return all test queries used by the parser test."""
    return retrieve_test_queries()


SAMPLE_DATASET = [
    {
        "id": "b1",
        "city": "Houston",
        "status": "ok",
        "model": {
            "damage_level": "destroyed",
            "confidence": 0.95,
            "reasoning": "Complete roof collapse and debris spread",
        },
        "evaluation": {
            "model_prediction": "destroyed",
            "ground_truth": "destroyed",
            "match": True,
        },
    },
    {
        "id": "b2",
        "city": "Houston",
        "status": "ok",
        "model": {
            "damage_level": "major-damage",
            "confidence": 0.82,
            "reasoning": "Severe structural damage",
        },
        "evaluation": {
            "model_prediction": "major-damage",
            "ground_truth": "minor-damage",
            "match": False,
        },
    },
    {
        "id": "b3",
        "city": "Miami",
        "status": "ok",
        "model": {
            "damage_level": "minor-damage",
            "confidence": 0.65,
            "reasoning": "Visible facade cracks",
        },
        "evaluation": {
            "model_prediction": "minor-damage",
            "ground_truth": "minor-damage",
            "match": True,
        },
    },
    {
        "id": "b4",
        "city": "Miami",
        "status": "failed",
        "model": {
            "damage_level": "destroyed",
            "confidence": 0.40,
            "reasoning": "Uncertain classification",
        },
        "evaluation": {
            "model_prediction": "destroyed",
            "ground_truth": "major-damage",
            "match": False,
        },
    },
    {
        "id": "b5",
        "city": "Dallas",
        "status": "ok",
        "model": {
            "damage_level": "no-damage",
            "confidence": 0.90,
            "reasoning": "No visible damage",
        },
        "evaluation": {
            "model_prediction": "no-damage",
            "ground_truth": "no-damage",
            "match": True,
        },
    },
]


@router.get("/integration-test")
def integration_test():
    """Full pipeline test using the sample dataset."""
    test_qp = QueryParser()
    test_qe = QueryEngine(SAMPLE_DATASET)
    test_dispatcher = IntentDispatcher(test_qe)

    test_cases = [
        {
            "query": "What damage was detected in Houston?",
            "expected_intent": "GET_DAMAGE_FOR_LOCATION",
            "expected_city": "Houston",
        },
        {
            "query": "How many buildings are destroyed?",
            "expected_intent": "GET_DAMAGE_DISTRIBUTION",
            "expected_damage": "destroyed",
        },
        {
            "query": "What is the model accuracy?",
            "expected_intent": "GET_MODEL_PERFORMANCE",
        },
        {"query": "Show me failed predictions", "expected_intent": "GET_FAILURE_CASES"},
        {
            "query": "Top 2 most damaged buildings",
            "expected_intent": "GET_TOP_K_DAMAGE",
            "expected_top_k": 2,
        },
        {
            "query": "Compare Houston and Dallas",
            "expected_intent": "COMPARE_LOCATIONS",
            "expected_cities": ["Houston", "Dallas"],
        },
        {
            "query": "Why was building b1 classified as destroyed?",
            "expected_intent": "GET_MODEL_EXPLANATION",
            "expected_id": "b1",
        },
        {
            "query": "Show confidence above 0.8",
            "expected_intent": "GET_CONFIDENCE_ANALYSIS",
            "expected_threshold": 0.8,
        },
        {
            "query": "Are there any issues in the dataset?",
            "expected_intent": "GET_DATASET_HEALTH",
        },
        {
            "query": "Compare Houston, El Paso, and Dallas",
            "expected_intent": "COMPARE_LOCATIONS",
            "expected_cities": ["Houston", "El Paso", "Dallas"],
        },  # fixed: include all three
        {"query": "Top buildings", "expected_intent": "GET_TOP_K_DAMAGE"},
        {"query": "What is the weather today?", "expected_intent": "OUT_OF_SCOPE"},
    ]

    results = []
    for test in test_cases:
        try:
            parsed = test_qp.parse(test["query"])
            result = test_dispatcher.dispatch(parsed)

            intent_match = parsed.get("intent") == test["expected_intent"]
            city_match = (
                parsed.get("city") == test.get("expected_city")
                if "expected_city" in test
                else True
            )
            damage_match = (
                parsed.get("damage_level") == test.get("expected_damage")
                if "expected_damage" in test
                else True
            )
            top_k_match = (
                parsed.get("top_k") == test.get("expected_top_k")
                if "expected_top_k" in test
                else True
            )
            id_match = (
                parsed.get("id") == test.get("expected_id")
                if "expected_id" in test
                else True
            )
            threshold_match = (
                parsed.get("confidence_threshold") == test.get("expected_threshold")
                if "expected_threshold" in test
                else True
            )

            # For city list comparison, we compare sets (order doesn't matter)
            if "expected_cities" in test:
                actual_cities = parsed.get("cities", [])
                city_match = set(actual_cities) == set(test["expected_cities"])

            valid_response = isinstance(result, dict) and "data" in result
            passed = all(
                [
                    intent_match,
                    city_match,
                    damage_match,
                    top_k_match,
                    id_match,
                    threshold_match,
                    valid_response,
                ]
            )

            results.append(
                {
                    "query": test["query"],
                    "expected_intent": test["expected_intent"],
                    "parsed_intent": parsed.get("intent"),
                    "intent_match": intent_match,
                    "city_match": city_match,
                    "damage_match": damage_match,
                    "top_k_match": top_k_match,
                    "id_match": id_match,
                    "threshold_match": threshold_match,
                    "valid_response": valid_response,
                    "passed": passed,
                    "parsed": parsed,
                    "result_preview": result.get("data", {}).get("type"),
                }
            )
        except Exception as e:
            results.append({"query": test["query"], "error": str(e), "passed": False})

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(passed / total, 3),
        "details": results,
    }


@router.get("/stress-test")
def stress_test():
    """Simple stress test by repeating a set of queries."""
    queries = [
        "Houston damage?",
        "Compare cities",
        "Top buildings",
        "Failures?",
        "Confidence low cases",
    ]
    results = []
    for q in queries * 2:
        parsed = qp.parse(q)
        result = dispatcher.dispatch(parsed)
        results.append(result["data"]["type"])
    return {"total_runs": len(results), "unique_outputs": list(set(results))}
