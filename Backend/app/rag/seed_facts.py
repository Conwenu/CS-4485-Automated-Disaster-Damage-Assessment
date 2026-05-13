"""Hand-curated authoritative answers for common definitional queries.

These take priority over vector retrieval. They guarantee correct, cited
answers for the questions users will ask most often, without any LLM call.
Sources: xBD paper (Gupta et al., 2019), FEMA Preliminary Damage Assessment
Guide (2025), and the Joint Damage Scale used by both.
"""

from typing import Dict, List, Optional, Tuple

# (matching keywords, answer, citation)
SEED_FACTS: List[Tuple[List[str], str, str]] = [
    (
        [
            "what does no-damage mean",
            "what is no-damage",
            "define no-damage",
            "no damage category",
            "what counts as no damage",
            "no visible damage definition",
            "no-damage definition",
        ],
        "In the Joint Damage Scale used by xBD and FEMA, 'no-damage' means the "
        "building shows no visible damage from the disaster. The structure is "
        "intact, with no visible cracks, no displaced debris, no standing water, "
        "and no signs of structural compromise in post-event imagery.",
        "xBD Joint Damage Scale (Gupta et al., 2019); FEMA PDA Guide (2025)",
    ),
    (
        [
            "what is minor damage",
            "what does minor-damage mean",
            "define minor damage",
            "minor damage category",
            "what counts as minor damage",
            "minor-damage definition",
            "minor damage level",
        ],
        "'Minor-damage' indicates the building is partially affected but largely "
        "functional. Typical signs include superficial or cosmetic damage, missing "
        "roof elements, visible cracks, water surrounding (but not entering) the "
        "structure, or volcanic flow nearby. The structure remains habitable "
        "without major repairs.",
        "xBD Joint Damage Scale (Gupta et al., 2019)",
    ),
    (
        [
            "what is major damage",
            "major-damage definition",
            "what does major-damage mean",
            "define major damage",
            "major damage category",
            "what counts as major damage",
            "major damage level",
        ],
        "'Major-damage' indicates substantial structural compromise. Typical signs "
        "include partial wall or roof collapse, the structure being surrounded by "
        "water or mud, water lines reaching between the first-floor windows and "
        "the roof, or encroaching volcanic flow. The building is not safely "
        "habitable without significant repair.",
        "xBD Joint Damage Scale (Gupta et al., 2019)",
    ),
    (
        [
            "what does destroyed mean",
            "define destroyed",
            "what is destroyed damage",
            "destroyed damage level",
            "what counts as destroyed",
            "destroyed category",
            "destroyed definition",
        ],
        "'Destroyed' means the structure is scorched, completely collapsed, "
        "covered with water or mud, or no longer present. Indicators include "
        "only the foundation remaining, two or more structural failures, threat "
        "of imminent collapse, or evidence of water reaching the roof or first "
        "ceiling. The building is a total loss.",
        "xBD Joint Damage Scale (Gupta et al., 2019)",
    ),
    (
        [
            "joint damage scale",
            "what is the joint damage scale",
            "damage scale used",
            "damage classification scheme",
            "ordinal damage scale",
            "xbd damage scale",
            "four damage categories",
            "damage levels in xbd",
        ],
        "The Joint Damage Scale used in xBD is a four-level ordinal scheme: "
        "0 = no-damage, 1 = minor-damage, 2 = major-damage, 3 = destroyed. It "
        "was created in collaboration with CAL FIRE, the California Air National "
        "Guard, and FEMA to unify damage reporting across disaster types. A single "
        "scale applies whether the building was burned, submerged, or collapsed.",
        "xBD paper (Gupta et al., 2019), Section 4",
    ),
    (
        [
            "how fema classifies",
            "fema damage assessment process",
            "what is a pda",
            "preliminary damage assessment",
            "how does fema classify",
            "fema classification process",
            "what does fema look for",
        ],
        "FEMA conducts Preliminary Damage Assessments (PDAs) after a disaster to "
        "determine eligibility for federal assistance. Inspectors categorize each "
        "structure as Destroyed, Major, Minor, Affected, or No Visible Damage, "
        "based on observable criteria such as structural failures, water-line "
        "height, roof damage, and debris. The classification feeds into decisions "
        "about Individual Assistance and Public Assistance funding.",
        "FEMA Preliminary Damage Assessment Guide (2025)",
    ),
    (
        [
            "why hard to classify",
            "why is the model bad at intermediate",
            "why minor and major are hard",
            "difficult to distinguish damage",
            "why does the model struggle",
            "intermediate damage hard",
            "why is major damage accuracy low",
        ],
        "Intermediate damage categories (minor and major) are notoriously hard to "
        "distinguish from overhead imagery. The visual gap between 'a damaged "
        "roof' (minor) and 'partial collapse' (major) is subtle in top-down views, "
        "and classifier performance on these middle classes is consistently lower "
        "than on the extremes throughout the xBD benchmark.",
        "xBD paper (Gupta et al., 2019); xView2 Challenge results",
    ),
]


def find_seed_fact(query: str) -> Optional[Dict[str, str]]:
    """Return the best-matching seed fact for the query, or None.

    Score = total length of matched keyword phrases. Longer phrase matches
    win over shorter ones, so 'major damage' beats 'damage' alone.
    """
    if not query:
        return None
    q = query.lower()
    best: Optional[Tuple[int, Dict[str, str]]] = None

    for keywords, answer, source in SEED_FACTS:
        score = sum(len(k) for k in keywords if k in q)
        if score > 0:
            candidate = {"answer": answer, "source": source}
            if best is None or score > best[0]:
                best = (score, candidate)

    return best[1] if best else None
