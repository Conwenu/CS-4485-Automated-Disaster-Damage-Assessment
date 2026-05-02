"""Response service.

Generates deterministic prose from structured dispatcher output by default.
An optional LLM "polish" pass (settings.USE_LLM_RESPONSE_POLISH=true) rewrites
for tone, constrained by a number-preservation guard that actually works.
"""
import re
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.services.suggestion_generator import SuggestionGenerator


class ResponseService:
    def __init__(self, query_engine=None):
        self.suggester = SuggestionGenerator()
        self._llm = None  # lazily created if polish is enabled

    def generate(
        self,
        query: str,
        parsed,  # ParsedQuery
        result: Dict[str, Any],
        external_info: Optional[str] = None,
        external_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = self._normalize(result.get("data") or {})
        base_text = self._format(data)

        if external_info:
            base_text = self._blend_external(base_text, external_info)
        elif external_note:
            base_text = f"{external_note}\n\n{base_text}"

        text = base_text
        if settings.USE_LLM_RESPONSE_POLISH:
            text = self._polish(query, base_text)

        suggestions = result.get("suggestions") or self.suggester.generate(parsed, data)

        return {
            "text": text,
            "suggestions": suggestions,
            "ui_actions": result.get("ui_actions", []),
        }

    @staticmethod
    def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return data
        if "stats" in data and "distribution" not in data:
            data["distribution"] = data["stats"]
        if "count" in data and "total" not in data:
            data["total"] = data["count"]
        return data
    
    @staticmethod
    def _blend_external(base: str, external: str) -> str:
        if not external:
            return base
        return f"{external.strip()}\n\n{base.strip()}"


    def _polish(self, query: str, base_text: str) -> str:
        try:
            if self._llm is None:
                # self._llm = ChatOpenAI(model=settings.MODEL_NAME,temperature=0, base_url=settings.OPENROUTER_BASE_URL, api_key=settings.OPENROUTER_API_KEY,)
                self._llm = ChatGoogleGenerativeAI(model=settings.GOOGLE_MODEL, api_key=settings.GOOGLE_API_KEY ,temperature=0)
                self._llm = ChatGoogleGenerativeAI(
                    model=settings.GOOGLE_MODEL,
                    temperature=0,
                    google_api_key=settings.GOOGLE_API_KEY,
                )
                
            prompt = ChatPromptTemplate.from_template(
                "You are rewriting an answer for readability.\n"
                "Reformat the answer in proper Markdown. Choose the best structure yourself—use headings, bullet lists, bold for key terms, code blocks for code, etc., wherever it improves clarity or scannability.\n"
                "STRICT RULES:\n"
                "- Do NOT change any numbers, percentages, or counts.\n"
                "- Do NOT add facts that aren't in the base text.\n"
                "- Keep it concise.\n\n"
                "User query: {query}\n"
                "Base answer (ground truth):\n{base}\n\n"
                "Return only the rewritten answer."
            )
            out = (prompt | self._llm).invoke({"query": query, "base": base_text})
            text = (getattr(out, "content", "") or "").strip()
            # Guard: every number from the base must survive the rewrite.
            if not self._numbers_preserved(base_text, text):
                return base_text
            return text or base_text
        except Exception:
            return base_text

    @staticmethod
    def _numbers_preserved(base: str, rewritten: str) -> bool:
        base_nums = set(re.findall(r"\d+\.?\d*", base))
        out_nums = set(re.findall(r"\d+\.?\d*", rewritten))
        return base_nums.issubset(out_nums)


    def _format(self, d: Dict[str, Any]) -> str:
        t = d.get("type")
        formatters = {
            "damage_summary": self._f_damage_summary,
            "distribution": self._f_distribution,
            "model_performance": self._f_model_performance,
            "failures": self._f_failures,
            "top_k": self._f_top_k,
            "comparison": self._f_comparison,
            "explanation": self._f_explanation,
            "confidence_analysis": self._f_confidence,
            "dataset_health": self._f_dataset_health,
            "building_details": self._f_building_details,
            "buildings_by_damage": self._f_buildings_by_damage,
            "confidence_outliers": self._f_confidence_outliers,
            "misclassifications": self._f_misclassifications,
            "accuracy_by_damage": self._f_accuracy_by_damage,
            "scene_summary": self._f_scene_summary,
            "highest_confidence": self._f_highest_confidence,
            "random_building": self._f_random_building,
            "building_comparison": self._f_building_comparison,
            "city_ranking": self._f_city_ranking,
            "filter_by_status": self._f_filter_by_status,
            "error": lambda x: x.get("message", "Something went wrong."),
            "out_of_scope": lambda x: x.get("message", "That's outside what I can answer."),
            "clarification_needed": lambda x: x.get("message", "I need more information."),
        }
        return formatters.get(t, lambda _: "I couldn't generate a response.")(d)

    @staticmethod
    def _city_label(d: Dict[str, Any]) -> str:
        return d.get("city_display") or d.get("city") or "the dataset"
    
    @staticmethod
    def _building_line(r: Dict[str, Any]) -> str:
        """Render a slimmed building record (id + predicted + confidence)."""
        bid = r.get("id", "?")
        pred = r.get("predicted") or "?"
        conf = r.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        return f"  - {bid}: predicted {pred} (confidence {conf_str})"


    @staticmethod
    def _misclass_line(r: Dict[str, Any]) -> str:
        """Render a misclassified building (predicted vs ground truth)."""
        bid = r.get("id", "?")
        pred = r.get("predicted") or "?"
        truth = r.get("ground_truth") or "?"
        return f"  - {bid}: predicted {pred}, actual {truth}"

    def _f_damage_summary(self, d):
        dist = d.get("distribution", {})
        total = d.get("total", 0)
        return (
            f"In {self._city_label(d)}, {total:,} buildings were analyzed. "
            f"{dist.get('destroyed', 0):,} were destroyed, "
            f"{dist.get('major-damage', 0):,} had major damage, "
            f"{dist.get('minor-damage', 0):,} had minor damage, "
            f"and {dist.get('no-damage', 0):,} showed no damage."
        )

    def _f_distribution(self, d):
        label = self._city_label(d) if d.get("city") else "the full dataset"
        dist = d.get("distribution", {})
        return (
            f"Damage distribution for {label}: "
            f"{dist.get('destroyed', 0)} destroyed, "
            f"{dist.get('major-damage', 0)} major, "
            f"{dist.get('minor-damage', 0)} minor, "
            f"{dist.get('no-damage', 0)} no damage."
        )

    def _f_model_performance(self, d):
        total = d.get("total", 0)
        if total == 0:
            return f"No evaluation data available for {self._city_label(d)}."
        return (
            f"In {self._city_label(d)}, the model achieved "
            f"{d.get('accuracy', 0) * 100:.2f}% accuracy over {total} evaluated buildings."
        )

    def _f_failures(self, d):
        count = d.get("count", 0)
        if count == 0:
            return f"No misclassifications found in {self._city_label(d)}."

        header = (
            f"The model made {count:,} incorrect predictions "
            f"in {self._city_label(d)}."
        )
        examples = (d.get("examples") or [])[:5]
        if examples:
            listing = "\n".join(self._misclass_line(e) for e in examples)
            return f"{header} Examples:\n{listing}"
        return header

    def _f_top_k(self, d):
        results = d.get("results") or []
        if not results:
            where = f" in {d['city_display']}" if d.get("city_display") else ""
            return f"No matching buildings found{where}."

        k = d.get("k", len(results))
        where = f" in {d['city_display']}" if d.get("city_display") else ""
        lvl = f" at damage level {d['damage_level']}" if d.get("damage_level") else ""

        header = f"Top {k} most severely damaged buildings{lvl}{where}:"
        lines = [self._building_line(r) for r in results]
        return header + "\n" + "\n".join(lines)

    def _f_comparison(self, d):
        lines = []
        for city, stats in (d.get("results") or {}).items():
            label = stats.get("city_display") or city
            if "error" in stats:
                lines.append(f"{label}: {stats['error']}")
                continue
            dist = stats.get("distribution", {})
            lines.append(
                f"{label}: {dist.get('destroyed', 0)} destroyed / "
                f"{stats.get('total', 0)} total"
            )
        return "Comparison:\n" + "\n".join(lines)

    def _f_explanation(self, d):
        return (
            f"Building {d.get('id')} was classified as {d.get('damage_level')} "
            f"with confidence {d.get('confidence', 0):.2f}. "
            f"Reason: {d.get('reasoning') or 'no reasoning recorded.'}"
        )

    def _f_confidence(self, d):
        avg = d.get("average_confidence")
        if avg is None:
            return "No confidence data available."
        return (
            f"Average model confidence in {self._city_label(d)} is "
            f"{avg:.2f} across {d.get('count', 0)} predictions."
        )

    def _f_dataset_health(self, d):
        return (
            f"The dataset contains {d.get('total', 0)} records, "
            f"of which {d.get('failed', 0)} failed processing."
        )

    def _f_building_details(self, d):
        r = d.get("record", {}) or {}
        m = r.get("model") or {}
        ev = r.get("evaluation") or {}
        return (
            f"Building {r.get('id')} in {r.get('city', 'unknown')}: "
            f"predicted {m.get('damage_level')} "
            f"(confidence {m.get('confidence', 0):.2f}), "
            f"ground truth {ev.get('ground_truth', 'N/A')}. "
            f"Model reasoning: {m.get('reasoning') or 'not recorded.'}"
        )

    def _f_buildings_by_damage(self, d):
        count = d.get("count", 0)
        if count == 0:
            return (
                f"Found 0 buildings with damage level "
                f"'{d.get('damage_level')}' in {self._city_label(d)}."
            )

        header = (
            f"Found {count:,} buildings with damage level "
            f"'{d.get('damage_level')}' in {self._city_label(d)}."
        )
        examples = (d.get("examples") or [])[:5]
        ids = (d.get("building_ids") or [])[:5]
        if examples:
            listing = "\n".join(self._building_line(e) for e in examples)
            return f"{header} First {len(examples)}:\n{listing}"
        if ids:
            return f"{header} First {len(ids)}: {', '.join(ids)}."
        return header

    def _f_confidence_outliers(self, d):
        count = d.get("count", 0)
        dirn = d.get("direction", "below")
        threshold = d.get("threshold", 0.5)
        if count == 0:
            return (
                f"No buildings with confidence {dirn} {threshold} "
                f"in {self._city_label(d)}."
            )

        header = (
            f"Found {count:,} buildings with confidence {dirn} {threshold} "
            f"in {self._city_label(d)}."
        )
        ids = (d.get("building_ids") or [])[:5]
        if ids:
            return f"{header} First {len(ids)}: {', '.join(ids)}."
        return header

    def _f_misclassifications(self, d):
        count = d.get("count", 0)
        lvl = d.get("damage_level")
        qualifier = f" of class {lvl}" if lvl else ""

        if count == 0:
            return f"No misclassifications{qualifier} found in {self._city_label(d)}."

        header = (
            f"The model made {count:,} incorrect predictions{qualifier} "
            f"in {self._city_label(d)}."
        )
        examples = (d.get("examples") or [])[:5]
        if examples:
            listing = "\n".join(self._misclass_line(e) for e in examples)
            return f"{header} Examples:\n{listing}"
        return header

    def _f_accuracy_by_damage(self, d):
        lvl = d.get("damage_level") or "all damage levels"
        total = d.get("total", 0)
        if total == 0:
            return f"No evaluation data available for '{lvl}'."
        return (
            f"For '{lvl}', the model achieved {d.get('accuracy', 0) * 100:.1f}% "
            f"accuracy over {total} evaluated buildings."
        )

    def _f_scene_summary(self, d):
        dist = d.get("distribution", {})
        return (
            f"Scene {d.get('scene_id')} in {d.get('city', 'unknown')}: "
            f"{d.get('total', 0)} buildings — "
            f"{dist.get('destroyed', 0)} destroyed, "
            f"{dist.get('major-damage', 0)} major, "
            f"{dist.get('minor-damage', 0)} minor, "
            f"{dist.get('no-damage', 0)} no damage."
        )

    def _f_highest_confidence(self, d):
        r = d.get("record", {}) or {}
        m = r.get("model") or {}
        return (
            f"Most-confident prediction: {r.get('id')} in {r.get('city', 'unknown')} "
            f"— {m.get('damage_level')} at confidence {m.get('confidence', 0):.2f}."
        )

    def _f_random_building(self, d):
        r = d.get("record", {}) or {}
        m = r.get("model") or {}
        return (
            f"Random pick: {r.get('id')} in {r.get('city', 'unknown')} "
            f"— predicted {m.get('damage_level')} at confidence {m.get('confidence', 0):.2f}."
        )

    def _f_building_comparison(self, d):
        lines = []
        for b in d.get("buildings", []) or []:
            m = b.get("model") or {}
            lines.append(
                f"{b.get('id')}: {m.get('damage_level')} "
                f"(confidence {m.get('confidence', 0):.2f})"
            )
        return "Comparison:\n" + "\n".join(lines)

    def _f_city_ranking(self, d):
        lines = []
        for r in d.get("ranking", []) or []:
            label = " ".join(
                part.capitalize() for part in r["city"].replace("-", " ").split()
            )
            lines.append(
                f"{label}: {r['destroyed']}/{r['total']} destroyed "
                f"({r['score'] * 100:.1f}%)"
            )
        if not lines:
            return "No ranking data available."
        return "Cities ranked by destruction:\n" + "\n".join(lines)

    def _f_filter_by_status(self, d):
        count = d.get("count", 0)
        status = d.get("status", "failed")
        if count == 0:
            return f"No buildings with status '{status}' in {self._city_label(d)}."

        header = (
            f"Found {count:,} buildings with status '{status}' "
            f"in {self._city_label(d)}."
        )
        ids = (d.get("building_ids") or [])[:5]
        if ids:
            return f"{header} First {len(ids)}: {', '.join(ids)}."
        return header