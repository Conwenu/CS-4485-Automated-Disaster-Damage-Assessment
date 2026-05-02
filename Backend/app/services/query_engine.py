"""Query engine: deterministic lookups over the JSON dataset.

City matching is layered:
  1. Exact match on canonical key
  2. Variant match (handles 'santa rosa' vs 'santa-rosa' vs 'santarosa')
  3. Fuzzy match for typos at threshold 0.85

Use resolve_city() at the start of any city-aware lookup. It returns the
canonical dataset key (e.g. 'santa-rosa') or None if no confident match.
"""
import difflib
import heapq
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


class QueryEngine:
    def __init__(self, dataset: List[Dict[str, Any]]):
        self.data = dataset

        self.by_city: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_damage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_id: Dict[str, Dict[str, Any]] = {}
        self.by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.city_stats: Dict[str, Dict[str, int]] = {}
        self.city_totals: Dict[str, int] = {}

        # Canonical city key (as stored in dataset) is the source of truth.
        # All variants map back to it.
        self._city_lookup: Dict[str, str] = {}

        for d in self.data:
            raw_city = d.get("city")
            model = d.get("model") or {}

            if raw_city:
                self.by_city[raw_city].append(d)
                for variant in self._city_variants(raw_city):
                    self._city_lookup[variant] = raw_city

            if d.get("id"):
                self.by_id[d["id"]] = d
            if model.get("damage_level"):
                self.by_damage[model["damage_level"]].append(d)

            scene_id = d.get("scene_id")
            if scene_id:
                self.by_scene[scene_id].append(d)

        for city, records in self.by_city.items():
            self.city_stats[city] = self.compute_distribution(records)
            self.city_totals[city] = len(records)

    # --------------------------------------------------------------
    # City matching
    # --------------------------------------------------------------
    @staticmethod
    def _city_variants(city: str) -> List[str]:
        """All canonical forms a user might type for a given dataset city.

        Lowercased, whitespace-collapsed; both hyphenated and spaced;
        plus a no-separator form to catch 'santarosa'.
        """
        base = re.sub(r"\s+", " ", city.strip().lower())
        return [
            base,
            base.replace("-", " "),
            base.replace(" ", "-"),
            base.replace("-", "").replace(" ", ""),
        ]

    def resolve_city(self, value: Optional[str]) -> Optional[str]:
        """Map a user-supplied city to the dataset's canonical key.

        Three-stage match:
          1. Exact (after normalization)
          2. Variant lookup (handles hyphen/space mismatches)
          3. Fuzzy match (typos), threshold 0.85.

        Returns the canonical dataset key (e.g. 'santa-rosa') or None.
        """
        if not value:
            return None

        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if not normalized:
            return None

        # Stages 1 + 2: try every variant of the user input
        for variant in self._city_variants(normalized):
            if variant in self._city_lookup:
                return self._city_lookup[variant]

        # Stage 3: fuzzy match against known variants
        candidates = list(self._city_lookup.keys())
        matches = difflib.get_close_matches(
            normalized, candidates, n=1, cutoff=0.85
        )
        if matches:
            return self._city_lookup[matches[0]]

        return None

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    @staticmethod
    def _damage_rank(level: Optional[str]) -> int:
        return {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3,
        }.get(level, -1)

    # --------------------------------------------------------------
    # Lookups
    # --------------------------------------------------------------
    def get_by_city(self, city: Optional[str]) -> List[Dict[str, Any]]:
        canonical = self.resolve_city(city)
        return self.by_city.get(canonical, []) if canonical else []

    def get_city_distribution(self, city: Optional[str]):
        canonical = self.resolve_city(city)
        return self.city_stats.get(canonical) if canonical else None

    def get_by_id(self, id_: Optional[str]):
        return self.by_id.get(id_) if id_ else None

    def get_all(self) -> List[Dict[str, Any]]:
        return self.data

    def known_cities(self) -> List[str]:
        """Canonical city keys present in the dataset."""
        return sorted(self.by_city.keys())

    # --------------------------------------------------------------
    # Aggregations
    # --------------------------------------------------------------
    def compute_distribution(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0}
        for r in records:
            lvl = (r.get("model") or {}).get("damage_level")
            if lvl in counts:
                counts[lvl] += 1
        return counts

    def compute_performance(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid = [r for r in records if r.get("evaluation")]
        total = len(valid)
        correct = sum(1 for r in valid if (r.get("evaluation") or {}).get("match") is True)
        return {
            "accuracy": round(correct / total, 4) if total else 0,
            "total": total,
        }

    def get_failures(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in records if (r.get("evaluation") or {}).get("match") is False]

    def top_k_damage(
        self,
        k: int,
        city: Optional[str] = None,
        damage_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data

        if damage_level:
            records = [
                r
                for r in records
                if (r.get("model") or {}).get("damage_level") == damage_level
            ]

        def sort_key(d):
            m = d.get("model") or {}
            return (self._damage_rank(m.get("damage_level")), m.get("confidence") or 0)

        return heapq.nlargest(k, records, key=sort_key)

    def confidence_analysis(
        self,
        records: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        confs = [
            (r.get("model") or {}).get("confidence")
            for r in records
            if (r.get("model") or {}).get("confidence") is not None
        ]
        if threshold is not None:
            confs = [c for c in confs if c >= threshold]
        if not confs:
            return {"average_confidence": None, "count": 0}
        return {
            "average_confidence": round(sum(confs) / len(confs), 4),
            "count": len(confs),
        }

    def dataset_health(self) -> Dict[str, int]:
        total = len(self.data)
        failed = sum(1 for d in self.data if d.get("status") != "ok")
        return {"total": total, "failed": failed}

    # --------------------------------------------------------------
    # Specialized queries
    # --------------------------------------------------------------
    def get_buildings_by_damage(
        self,
        city: Optional[str],
        damage_level: str,
    ) -> List[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data
        return [
            r
            for r in records
            if (r.get("model") or {}).get("damage_level") == damage_level
        ]

    def get_confidence_outliers(
        self,
        city: Optional[str],
        threshold: float,
        direction: str,
    ) -> List[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data

        if direction == "below":
            return [
                r for r in records
                if (r.get("model") or {}).get("confidence", 1.0) < threshold
            ]
        return [
            r for r in records
            if (r.get("model") or {}).get("confidence", 0.0) > threshold
        ]

    def get_misclassifications(
        self,
        city: Optional[str],
    ) -> List[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data
        return [r for r in records if (r.get("evaluation") or {}).get("match") is False]

    def get_accuracy_by_damage(
        self,
        city: Optional[str],
        damage_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data

        if damage_level:
            records = [
                r
                for r in records
                if (r.get("model") or {}).get("damage_level") == damage_level
            ]
        valid = [r for r in records if r.get("evaluation")]
        total = len(valid)
        correct = sum(1 for r in valid if (r.get("evaluation") or {}).get("match") is True)
        return {
            "accuracy": round(correct / total, 4) if total else 0,
            "total": total,
            "damage_level": damage_level,
        }

    def get_scene_summary(self, scene_id: str) -> Optional[Dict[str, Any]]:
        records = self.by_scene.get(scene_id, [])
        if not records:
            return None
        return {
            "scene_id": scene_id,
            "city": records[0].get("city"),
            "total": len(records),
            "distribution": self.compute_distribution(records),
        }

    def rank_cities_by_damage(
        self,
        metric: str = "destroyed_percentage",
    ) -> List[Dict[str, Any]]:
        out = []
        for city, records in self.by_city.items():
            total = len(records)
            destroyed = self.city_stats.get(city, {}).get("destroyed", 0)
            if metric == "destroyed_percentage" and total:
                score = destroyed / total
            else:
                score = destroyed
            out.append(
                {
                    "city": city,
                    "total": total,
                    "destroyed": destroyed,
                    "score": round(score, 4),
                }
            )
        return sorted(out, key=lambda x: x["score"], reverse=True)

    def get_highest_confidence(self, city: Optional[str]) -> Optional[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data
        if not records:
            return None
        return max(records, key=lambda r: (r.get("model") or {}).get("confidence", 0))

    def get_lowest_confidence(self, city: Optional[str]) -> Optional[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data
        if not records:
            return None
        return min(records, key=lambda r: (r.get("model") or {}).get("confidence", 1))

    def get_random_building(
        self,
        city: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if scene_id:
            records = self.by_scene.get(scene_id, [])
        elif city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data
        return random.choice(records) if records else None

    def filter_by_status(
        self,
        city: Optional[str],
        status: str,
    ) -> List[Dict[str, Any]]:
        if city:
            canonical = self.resolve_city(city)
            records = self.by_city.get(canonical, []) if canonical else []
        else:
            records = self.data
        return [r for r in records if r.get("status") == status]