"""Intent dispatcher: maps parsed intent + params to engine calls.

Responsibilities:
- Validate that required parameters are present; if not, build a
  pending_clarification descriptor and return a clarification_needed result.
- Route to the correct engine method.
- Attach UI actions and starter suggestions to every response.
"""
from typing import Dict, Any, Optional
from app.services.suggestion_generator import SuggestionGenerator


class IntentDispatcher:
    def __init__(self, query_engine):
        self.qe = query_engine
        self._suggester = SuggestionGenerator(query_engine=query_engine)
        self.routes = {
            "GET_DAMAGE_FOR_LOCATION": self._handle_damage_for_location,
            "GET_DAMAGE_DISTRIBUTION": self._handle_damage_distribution,
            "GET_MODEL_PERFORMANCE": self._handle_model_performance,
            "GET_FAILURE_CASES": self._handle_failure_cases,
            "GET_TOP_K_DAMAGE": self._handle_top_k_damage,
            "COMPARE_LOCATIONS": self._handle_compare_locations,
            "GET_MODEL_EXPLANATION": self._handle_model_explanation,
            "GET_CONFIDENCE_ANALYSIS": self._handle_confidence_analysis,
            "GET_DATASET_HEALTH": self._handle_dataset_health,
            "GET_BUILDING_DETAILS": self._handle_building_details,
            "GET_BUILDINGS_BY_DAMAGE": self._handle_buildings_by_damage,
            "GET_CONFIDENCE_OUTLIERS": self._handle_confidence_outliers,
            "GET_MISCLASSIFICATIONS": self._handle_misclassifications,
            "GET_ACCURACY_BY_DAMAGE": self._handle_accuracy_by_damage,
            "GET_SCENE_SUMMARY": self._handle_scene_summary,
            "GET_HIGHEST_CONFIDENCE": self._handle_highest_confidence,
            "GET_RANDOM_BUILDING": self._handle_random_building,
            "COMPARE_BUILDINGS": self._handle_compare_buildings,
            "RANK_CITIES_BY_DAMAGE": self._handle_rank_cities,
            "FILTER_BY_STATUS": self._handle_filter_by_status,
        }

    def dispatch(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        intent = parsed.get("intent")

        err = self._validate(intent, parsed)
        if err:
            pending = self._build_pending(intent, parsed)
            return self._wrap(
                {
                    "type": "clarification_needed",
                    "message": err,
                    "pending_clarification": pending,
                }
            )

        handler = self.routes.get(intent, self._handle_out_of_scope)
        return self._wrap(handler(parsed))

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------
    def _validate(self, intent: str, q: Dict[str, Any]) -> Optional[str]:
        city_required = {
            "GET_DAMAGE_FOR_LOCATION": "Which city are you asking about?",
            "GET_MODEL_PERFORMANCE": "Which city's model performance?",
            "GET_FAILURE_CASES": "Which city?",
            "GET_CONFIDENCE_ANALYSIS": "Which city?",
        }
        if intent in city_required and not q.get("city"):
            return city_required[intent]

        if intent == "COMPARE_LOCATIONS":
            if len(q.get("cities") or []) < 2:
                return "Please name at least two cities to compare."

        if intent == "GET_TOP_K_DAMAGE" and q.get("top_k") is None:
            return "How many buildings would you like to see?"

        if intent == "GET_MODEL_EXPLANATION" and not q.get("id"):
            return "Which building ID should I explain?"

        if intent == "GET_BUILDING_DETAILS" and not q.get("id"):
            return "Which building ID?"

        if intent == "GET_BUILDINGS_BY_DAMAGE" and not q.get("damage_level"):
            return "Which damage level? (no-damage, minor-damage, major-damage, destroyed)"

        if intent == "GET_SCENE_SUMMARY" and not q.get("scene_id"):
            return "Please provide a scene ID."

        if intent == "COMPARE_BUILDINGS":
            if len(q.get("ids") or []) < 2:
                return "Please name at least two building IDs to compare."

        if intent == "GET_CONFIDENCE_OUTLIERS" and q.get("confidence_threshold") is None:
            return "What confidence threshold (between 0 and 1)?"

        if intent == "FILTER_BY_STATUS" and not q.get("status"):
            return "Which status — 'ok' or 'failed'?"

        return None

    def _build_pending(self, intent: str, q: Dict[str, Any]) -> Dict[str, Any]:
        # Carry forward everything the user already gave so we only re-ask for
        # what's missing.
        carried = {
            k: v
            for k, v in q.items()
            if k
            in (
                "city",
                "cities",
                "id",
                "ids",
                "scene_id",
                "top_k",
                "damage_level",
                "confidence_threshold",
                "direction",
                "status",
            )
            and v not in (None, [], "")
        }

        missing_map = {
            "GET_DAMAGE_FOR_LOCATION": "city",
            "GET_MODEL_PERFORMANCE": "city",
            "GET_FAILURE_CASES": "city",
            "GET_CONFIDENCE_ANALYSIS": "city",
            "COMPARE_LOCATIONS": "cities",
            "GET_TOP_K_DAMAGE": "top_k",
            "GET_MODEL_EXPLANATION": "id",
            "GET_BUILDING_DETAILS": "id",
            "GET_BUILDINGS_BY_DAMAGE": "damage_level",
            "GET_SCENE_SUMMARY": "scene_id",
            "COMPARE_BUILDINGS": "ids",
            "GET_CONFIDENCE_OUTLIERS": "confidence_threshold",
            "FILTER_BY_STATUS": "status",
        }
        return {
            "intent": intent,
            "missing_param": missing_map.get(intent),
            "params": carried,
        }
        
    def _no_city_data_error(self, city: str) -> Dict[str, Any]:
        """Build a friendly error when a user-supplied city isn't in the dataset."""
        known = self.qe.known_cities()
        if known:
            display = ", ".join(self._display(c) for c in known[:5])
            extra = "" if len(known) <= 5 else f" (and {len(known) - 5} more)"
            return {
                "type": "error",
                "message": (
                    f"I don't have data for '{city}'. The dataset currently "
                    f"covers: {display}{extra}."
                ),
            }
        return {"type": "error", "message": f"No data found for {city}."}


    def _handle_damage_for_location(self, q):
        city = q.get("city")
        canonical = self.qe.resolve_city(city)
        if not canonical:
            return self._no_city_data_error(city)

        records = self.qe.by_city.get(canonical, [])
        if not records:
            return self._no_city_data_error(city)

        stats = (
            self.qe.get_city_distribution(canonical)
            or self.qe.compute_distribution(records)
        )
        return {
            "type": "damage_summary",
            "city": canonical,
            "city_display": self._display(canonical),
            "total": len(records),
            "distribution": stats,
        }

    def _handle_damage_distribution(self, q):
        city = q.get("city")
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)
            records = self.qe.by_city.get(canonical, [])
            return {
                "type": "distribution",
                "city": canonical,
                "city_display": self._display(canonical),
                "distribution": self.qe.compute_distribution(records),
                "total": len(records),
            }

        records = self.qe.get_all()
        return {
            "type": "distribution",
            "city": None,
            "city_display": None,
            "distribution": self.qe.compute_distribution(records),
            "total": len(records),
        }

    def _handle_model_performance(self, q):
        city = q.get("city")
        canonical = self.qe.resolve_city(city)
        if not canonical:
            return self._no_city_data_error(city)

        records = self.qe.by_city.get(canonical, [])
        if not records:
            return self._no_city_data_error(city)

        return {
            "type": "model_performance",
            "city": canonical,
            "city_display": self._display(canonical),
            **self.qe.compute_performance(records),
        }

    def _handle_failure_cases(self, q):
        city = q.get("city")
        canonical = self.qe.resolve_city(city)
        if not canonical:
            return self._no_city_data_error(city)

        records = self.qe.by_city.get(canonical, [])
        failures = self.qe.get_failures(records) if records else []
        return {
            "type": "failures",
            "city": canonical,
            "city_display": self._display(canonical),
            "count": len(failures),
            "examples": [self._slim(r) for r in failures[:5]],
        }

    def _handle_top_k_damage(self, q):
        k = q.get("top_k") or 5
        city = q.get("city")
        damage_level = q.get("damage_level")

        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        results = self.qe.top_k_damage(k, city=canonical, damage_level=damage_level)
        return {
            "type": "top_k",
            "k": k,
            "city": canonical,
            "city_display": self._display(canonical) if canonical else None,
            "damage_level": damage_level,
            "results": [self._slim(r) for r in results],
            "building_ids": [r["id"] for r in results],
        }

    def _handle_compare_locations(self, q):
        cities = q.get("cities") or []
        results = {}
        for city in cities:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                results[city] = {"error": f"No data found for {city}."}
                continue

            records = self.qe.by_city.get(canonical, [])
            if not records:
                results[city] = {"error": f"No data found for {city}."}
                continue

            stats = (
                self.qe.get_city_distribution(canonical)
                or self.qe.compute_distribution(records)
            )
            results[canonical] = {
                "total": len(records),
                "distribution": stats,
                "city_display": self._display(canonical),
            }
        return {"type": "comparison", "results": results}
    
    
    
    
    
    

    def _handle_model_explanation(self, q):
        rec = self.qe.get_by_id(q.get("id"))
        if not rec:
            return {"type": "error", "message": f"Building {q.get('id')} not found."}
        m = rec.get("model") or {}
        return {
            "type": "explanation",
            "id": rec.get("id"),
            "damage_level": m.get("damage_level"),
            "confidence": m.get("confidence"),
            "reasoning": m.get("reasoning"),
        }

    def _handle_confidence_analysis(self, q):
        city = q.get("city")
        canonical = self.qe.resolve_city(city)
        if not canonical:
            return self._no_city_data_error(city)

        records = self.qe.by_city.get(canonical, [])
        if not records:
            return {
                "type": "confidence_analysis",
                "average_confidence": None,
                "count": 0,
                "city": canonical,
                "city_display": self._display(canonical),
            }
        return {
            "type": "confidence_analysis",
            "city": canonical,
            "city_display": self._display(canonical),
            **self.qe.confidence_analysis(records, q.get("confidence_threshold")),
        }

    def _handle_dataset_health(self, q):
        return {"type": "dataset_health", **self.qe.dataset_health()}

    def _handle_building_details(self, q):
        rec = self.qe.get_by_id(q.get("id"))
        if not rec:
            return {"type": "error", "message": f"Building {q.get('id')} not found."}
        return {"type": "building_details", "record": rec}

    def _handle_buildings_by_damage(self, q):
        city = q.get("city")
        damage_level = q.get("damage_level")

        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        records = self.qe.get_buildings_by_damage(canonical, damage_level)
        return {
            "type": "buildings_by_damage",
            "city": canonical,
            "city_display": self._display(canonical) if canonical else None,
            "damage_level": damage_level,
            "count": len(records),
            "building_ids": [r["id"] for r in records],
        }
        
    def _handle_confidence_outliers(self, q):
        city = q.get("city")
        threshold = q.get("confidence_threshold", 0.5)
        direction = q.get("direction") or "below"

        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        records = self.qe.get_confidence_outliers(canonical, threshold, direction)
        return {
            "type": "confidence_outliers",
            "city": canonical,
            "city_display": self._display(canonical) if canonical else None,
            "threshold": threshold,
            "direction": direction,
            "count": len(records),
            "building_ids": [r["id"] for r in records],
        }

    def _handle_misclassifications(self, q):
        city = q.get("city")
        damage_level = q.get("damage_level")

        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        records = self.qe.get_misclassifications(canonical)

        # Optional damage-level filter, applied to ground truth
        # (so 'minor-damage misses' = buildings actually minor-damage that
        # the model got wrong).
        if damage_level:
            records = [
                r for r in records
                if (r.get("evaluation") or {}).get("ground_truth") == damage_level
            ]

        return {
            "type": "misclassifications",
            "city": canonical,
            "city_display": self._display(canonical) if canonical else None,
            "damage_level": damage_level,
            "count": len(records),
            "examples": [self._slim(r) for r in records[:5]],
        }

    def _handle_accuracy_by_damage(self, q):
        city = q.get("city")
        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        return {
            "type": "accuracy_by_damage",
            "city": canonical,
            **self.qe.get_accuracy_by_damage(canonical, q.get("damage_level")),
        }


    def _handle_scene_summary(self, q):
        summary = self.qe.get_scene_summary(q.get("scene_id"))
        if not summary:
            return {"type": "error", "message": f"Scene {q.get('scene_id')} not found."}
        return {"type": "scene_summary", **summary}

    def _handle_highest_confidence(self, q):
        city = q.get("city")
        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        rec = self.qe.get_highest_confidence(canonical)
        if not rec:
            return {"type": "error", "message": "No buildings found."}
        return {"type": "highest_confidence", "record": rec}

    def _handle_random_building(self, q):
        city = q.get("city")
        scene_id = q.get("scene_id")

        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        rec = self.qe.get_random_building(city=canonical, scene_id=scene_id)
        if not rec:
            return {"type": "error", "message": "No buildings found."}
        return {"type": "random_building", "record": rec}

    def _handle_compare_buildings(self, q):
        ids = q.get("ids") or []
        records = [r for r in (self.qe.get_by_id(i) for i in ids) if r]
        if not records:
            return {"type": "error", "message": "None of the provided building IDs were found."}
        return {"type": "building_comparison", "buildings": records}

    def _handle_rank_cities(self, q):
        return {"type": "city_ranking", "ranking": self.qe.rank_cities_by_damage()[:5]}

    def _handle_filter_by_status(self, q):
        city = q.get("city")
        status = q.get("status", "failed")

        canonical = None
        if city:
            canonical = self.qe.resolve_city(city)
            if not canonical:
                return self._no_city_data_error(city)

        records = self.qe.filter_by_status(canonical, status)
        return {
            "type": "filter_by_status",
            "city": canonical,
            "city_display": self._display(canonical) if canonical else None,
            "status": status,
            "count": len(records),
            "building_ids": [r["id"] for r in records],
        }

    def _handle_out_of_scope(self, q):
        return {
            "type": "out_of_scope",
            "message": (
                "I can only answer questions about the disaster damage dataset — "
                "damage levels, model predictions, accuracy, cities or buildings, "
                "and related disaster concepts. Try asking about a city, a damage "
                "level, or the model's performance."
            ),
        }

    # -------------------------------------------------------------
    # Wrapping / suggestions / UI actions
    # -------------------------------------------------------------
    def _wrap(self, result: Dict[str, Any]) -> Dict[str, Any]:
        wrapped = {
            "data": result,
            "suggestions": self._suggest(result),
            "ui_actions": self._ui_actions(result),
        }
        if result.get("pending_clarification"):
            wrapped["pending_clarification"] = result["pending_clarification"]
        return wrapped

    @staticmethod
    def _display(city_key: Optional[str]) -> Optional[str]:
        if not city_key:
            return None
        return " ".join(
            part.capitalize() for part in city_key.replace("-", " ").split()
        )

    @staticmethod
    def _slim(record: Dict[str, Any]) -> Dict[str, Any]:
        m = record.get("model") or {}
        ev = record.get("evaluation") or {}
        return {
            "id": record.get("id"),
            "city": record.get("city"),
            "scene_id": record.get("scene_id"),
            "predicted": m.get("damage_level"),
            "ground_truth": ev.get("ground_truth"),
            "match": ev.get("match"),
            "confidence": m.get("confidence"),
        }

    def _suggest(self, result: Dict[str, Any]):
        return self._suggester.generate(parsed=None, data=result)

    def _ui_actions(self, result: Dict[str, Any]):
        """UI action vocabulary. Your frontend teammate can extend this as the
        map wiring is finalized (add bounds/colors to FOCUS_CITY, etc.).
        """
        t = result.get("type")
        if t == "damage_summary":
            return [{"type": "FOCUS_CITY", "city": result.get("city")}]
        if t == "distribution" and result.get("city"):
            return [{"type": "FOCUS_CITY", "city": result.get("city")}]
        if t in ("top_k", "buildings_by_damage", "confidence_outliers", "filter_by_status"):
            return [
                {
                    "type": "HIGHLIGHT_BUILDINGS",
                    "building_ids": result.get("building_ids", []),
                    "color_by": "damage_level",
                }
            ]
        if t == "building_details":
            rec = result.get("record", {}) or {}
            return [{"type": "HIGHLIGHT_BUILDINGS", "building_ids": [rec.get("id")]}]
        if t == "scene_summary":
            return [{"type": "FOCUS_SCENE", "scene_id": result.get("scene_id")}]
        return []