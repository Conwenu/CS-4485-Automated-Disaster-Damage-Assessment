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
        ["no-damage", "no damage", "undamaged", "intact", "no visible damage"],
        "In the Joint Damage Scale used by xBD and FEMA, 'no-damage' means the "
        "building shows no visible damage from the disaster. The structure is "
        "intact, with no visible cracks, no displaced debris, no standing water, "
        "and no signs of structural compromise in post-event imagery.",
        "xBD Joint Damage Scale (Gupta et al., 2019); FEMA PDA Guide (2025)",
    ),
    (
        ["minor-damage", "minor damage", "lightly damaged"],
        "'Minor-damage' indicates the building is partially affected but largely "
        "functional. Typical signs include superficial or cosmetic damage, missing "
        "roof elements, visible cracks, water surrounding (but not entering) the "
        "structure, or volcanic flow nearby. The structure remains habitable "
        "without major repairs.",
        "xBD Joint Damage Scale (Gupta et al., 2019)",
    ),
    (
        ["major-damage", "major damage", "heavily damaged", "significant damage"],
        "'Major-damage' indicates substantial structural compromise. Typical signs "
        "include partial wall or roof collapse, the structure being surrounded by "
        "water or mud, water lines reaching between the first-floor windows and "
        "the roof, or encroaching volcanic flow. The building is not safely "
        "habitable without significant repair.",
        "xBD Joint Damage Scale (Gupta et al., 2019)",
    ),
    (
        ["destroyed", "totaled", "leveled", "wiped out", "completely destroyed"],
        "'Destroyed' means the structure is scorched, completely collapsed, "
        "covered with water or mud, or no longer present. Indicators include "
        "only the foundation remaining, two or more structural failures, threat "
        "of imminent collapse, or evidence of water reaching the roof or first "
        "ceiling. The building is a total loss.",
        "xBD Joint Damage Scale (Gupta et al., 2019)",
    ),
    (
        ["joint damage scale", "damage scale", "damage categories",
         "classification scheme", "ordinal scale"],
        "The Joint Damage Scale used in xBD is a four-level ordinal scheme: "
        "0 = no-damage, 1 = minor-damage, 2 = major-damage, 3 = destroyed. It "
        "was created in collaboration with CAL FIRE, the California Air National "
        "Guard, and FEMA to unify damage reporting across disaster types. A single "
        "scale applies whether the building was burned, submerged, or collapsed.",
        "xBD paper (Gupta et al., 2019), Section 4",
    ),
    (
        ["fema", "how fema classifies", "fema damage assessment", "pda",
         "preliminary damage assessment"],
        "FEMA conducts Preliminary Damage Assessments (PDAs) after a disaster to "
        "determine eligibility for federal assistance. Inspectors categorize each "
        "structure as Destroyed, Major, Minor, Affected, or No Visible Damage, "
        "based on observable criteria such as structural failures, water-line "
        "height, roof damage, and debris. The classification feeds into decisions "
        "about Individual Assistance and Public Assistance funding.",
        "FEMA Preliminary Damage Assessment Guide (2025)",
    ),
    (
        ["why hard", "hard to classify", "difficult to distinguish",
         "why is the model bad", "why minor and major"],
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