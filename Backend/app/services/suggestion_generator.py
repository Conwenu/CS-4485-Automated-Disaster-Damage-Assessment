"""Deterministic suggestion generator. Zero latency, zero hallucination risk.

Suggestions are dataset-aware: the generator checks how many cities the
engine knows about and avoids dead-end follow-ups (e.g. won't suggest
"compare with another city" when there's only one city in the dataset).
"""

from typing import Any, Dict, List, Optional


class SuggestionGenerator:
    def __init__(self, query_engine=None) -> None:
        self.qe = query_engine

    def generate(self, parsed, data: Dict[str, Any]) -> List[str]:
        if data is None:
            data = {}
        t = data.get("type")
        city = data.get("city_display") or data.get("city")

        if t == "damage_summary" and city:
            return self._for_damage_summary(city)
        if t == "distribution":
            return self._for_distribution(city)
        if t == "comparison":
            return self._for_comparison(data)
        if t == "confidence_analysis" and city:
            return self._for_confidence_analysis(city)
        if t == "buildings_by_damage":
            return self._for_buildings_by_damage(city, data.get("damage_level"))
        if t == "misclassifications":
            return self._for_misclassifications(city)
        if t == "building_details":
            return self._for_building_details(data)
        if t == "top_k":
            return self._for_top_k(city)
        if t == "city_ranking":
            return self._for_city_ranking()
        if t == "model_performance" and city:
            return self._for_model_performance(city)
        if t == "scene_summary":
            return self._for_scene_summary(data)
        if t == "highest_confidence":
            return self._for_highest_confidence(data)
        if t == "random_building":
            return self._for_random_building(data)
        if t == "out_of_scope":
            return self._generic()

        return self._generic()

    def _multi_city(self) -> bool:
        return self.qe is not None and len(self.qe.known_cities()) > 1

    def _other_cities(self, exclude: Optional[str]) -> List[str]:
        """Cities other than the one in the current result. Display-formatted."""
        if self.qe is None:
            return []
        canonical_excluded = self.qe.resolve_city(exclude) if exclude else None
        out: List[str] = []
        for c in self.qe.known_cities():
            if c == canonical_excluded:
                continue
            out.append(self._display(c))
        return out

    @staticmethod
    def _display(city_key: str) -> str:
        return " ".join(
            part.capitalize() for part in city_key.replace("-", " ").split()
        )

    def _for_damage_summary(self, city: str) -> List[str]:
        suggestions = [
            f"Show the top 5 most damaged buildings in {city}",
            f"What's the model's accuracy in {city}?",
        ]
        others = self._other_cities(city)
        if others:
            suggestions.append(f"Compare {city} with {others[0]}")
        else:
            # Single-city dataset — push deeper into this city instead
            suggestions.append(f"Where did the model misclassify in {city}?")
        return suggestions

    def _for_distribution(self, city: Optional[str]) -> List[str]:
        if city:
            return [
                f"Show the top 5 most damaged buildings in {city}",
                f"How accurate is the model in {city}?",
                "Show misclassifications",
            ]
        # Whole-dataset distribution
        suggestions = ["How accurate is the model overall?"]
        if self._multi_city():
            suggestions.append("Which city was hit hardest?")
        else:
            suggestions.append("Show me 5 destroyed buildings")
        suggestions.append("Show accuracy by damage level")
        return suggestions

    def _for_comparison(self, data: Dict[str, Any]) -> List[str]:
        cities = list((data.get("results") or {}).keys())
        first = self._display(cities[0]) if cities else "a city"
        return [
            f"Show top damaged buildings in {first}",
            "How accurate is the model in each?",
            "Which of these has the most destroyed?",
        ]

    def _for_confidence_analysis(self, city: str) -> List[str]:
        return [
            f"Show predictions with confidence below 0.6 in {city}",
            f"Where did the model go wrong in {city}?",
            f"What's the model's accuracy in {city}?",
        ]

    def _for_buildings_by_damage(
        self, city: Optional[str], damage_level: Optional[str]
    ) -> List[str]:
        suffix = f" in {city}" if city else ""
        suggestions = [f"Show a random building{suffix}"]
        if damage_level:
            suggestions.append(f"Show accuracy for {damage_level} buildings")
            suggestions.append(f"Show misclassified {damage_level} buildings")
        else:
            suggestions.append("Show accuracy by damage level")
            suggestions.append("Show misclassifications")
        return suggestions

    def _for_misclassifications(self, city: Optional[str]) -> List[str]:
        return [
            "Show accuracy by damage level",
            "Show low-confidence predictions",
            f"Show failure cases in {city}" if city else "Show dataset health",
        ]

    def _for_building_details(self, data: Dict[str, Any]) -> List[str]:
        rec = data.get("record") or {}
        scene_id = rec.get("scene_id")
        suggestions = ["Explain why this was classified that way"]
        if scene_id:
            suggestions.append(f"Show the rest of scene {scene_id}")
        suggestions.append("Show another random building")
        return suggestions

    def _for_top_k(self, city: Optional[str]) -> List[str]:
        return [
            "How accurate is the model overall?",
            "Show misclassifications",
            f"Pick a random building in {city}" if city else "Pick a random building",
        ]

    def _for_city_ranking(self) -> List[str]:
        return [
            "Show damage distribution for the worst-hit city",
            "Compare the top two cities",
            "Show model accuracy by city",
        ]

    def _for_model_performance(self, city: str) -> List[str]:
        return [
            f"Show accuracy by damage level in {city}",
            f"Where did the model go wrong in {city}?",
            f"Show low-confidence predictions in {city}",
        ]

    def _for_scene_summary(self, data: Dict[str, Any]) -> List[str]:
        scene_id = data.get("scene_id")
        suggestions = ["Pick a random building from this scene"]
        if scene_id:
            suggestions.append(f"Show destroyed buildings in scene {scene_id}")
        suggestions.append("Show accuracy by damage level")
        return suggestions

    def _for_highest_confidence(self, data: Dict[str, Any]) -> List[str]:
        return [
            "Explain why this was classified that way",
            "Show low-confidence predictions",
            "Show another random building",
        ]

    def _for_random_building(self, data: Dict[str, Any]) -> List[str]:
        return [
            "Explain why this was classified that way",
            "Show another random building",
            "Show details for another building",
        ]

    def _generic(self) -> List[str]:
        suggestions = ["Show the overall damage distribution"]
        if self._multi_city():
            suggestions.append("Which city was hit hardest?")
        else:
            suggestions.append("Show accuracy by damage level")
        suggestions.append("How accurate is the model?")
        return suggestions
