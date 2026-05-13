import json
from typing import List, Dict, Optional, Any
from pathlib import Path


def get_damage_records():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "temp_data.json"

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_new_damage_records():
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "damage_dataset.json"

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_damage_data(query: str, limit: Optional[int] = 10) -> List[Dict]:
    """
    Search for damage records by city name.
    """
    DAMAGE_RECORDS = get_damage_records()

    if not query or not isinstance(query, str):
        return []

    q = query.lower()

    results = []

    for record in DAMAGE_RECORDS:
        city = str(record.get("city") or "").lower()

        model = record.get("model") or {}
        damage = str(model.get("damage_level") or "").lower()

        score = 0

        if city in q:
            score += 2

        if damage in q:
            score += 2

        if score > 0:
            results.append((score, record))

    results.sort(key=lambda x: x[0], reverse=True)

    results = [r for _, r in results]

    if limit:
        results = results[:limit]

    return results


def format_context(records: List[Dict]) -> str:
    """
    Format damage records into a human-readable string.
    """
    if not records:
        return "No damage records found for this location."

    lines = ["Damage assessment data:"]
    for r in records:
        city = r.get("city", "Unknown city")
        status = r.get("status", "unknown")
        model = r.get("model", {})
        damage_level = model.get("damage_level", "unknown")
        confidence = model.get("confidence", None)

        # Basic info line
        line = f"- {city}: damage level = {damage_level}"
        if confidence is not None:
            line += f" (confidence: {confidence:.2f})"
        else:
            line += " (confidence: N/A)"

        if status != "ok":
            line += f" [status: {status}]"

        lines.append(line)

        # should include model reasoning when available
        reasoning = model.get("reasoning")
        if reasoning:
            lines.append(f"    Reasoning: {reasoning}")

    return "\n".join(lines)


def format_relevant_data(records: List[Dict]) -> List[Dict]:
    cleaned = []

    for r in records:
        model = r.get("model", {})

        cleaned.append(
            {
                "id": r.get("id"),
                "city": r.get("city"),
                "damage_level": model.get("damage_level"),
                "confidence": model.get("confidence"),
                "status": r.get("status"),
            }
        )

    return cleaned


def retrieve_test_queries():
    # TODO:
    # Instead of reading this file every time the function is called,
    # load it once when the FastAPI app starts and store it in app.state. r to just load it up on startup

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "test_queries.json"

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_true_data() -> List[Dict[str, Any]]:
    file_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "santa_rosa"
        / "building_results.json"
    )

    if not file_path.exists():
        print(f"[data loader] No dataset found at {file_path!r}; returning empty list.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: object = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[data loader] Failed to load JSON: {e}; returning empty list.")
        return []

    # Type narrowing / validation
    if not isinstance(data, list):
        print("[data loader] Expected a list; returning empty list.")
        return []

    if not all(isinstance(item, dict) for item in data):
        print("[data loader] Expected list of dicts; returning empty list.")
        return []

    return data
