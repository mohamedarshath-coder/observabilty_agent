"""Regex-only PII/financial redaction, mirroring lib/piiMasking.js in the Node backend.

Kept as a separate Python copy (not a shared module — different language/process) for the
same reason server.js's own masking exists: Langfuse's `mask_otel_spans` export hook must
be synchronous, so this can't call out to an LLM or run the Node side's async NER pass.
Financial regex + known/common-name patterns only. Keep in sync with lib/piiMasking.js if
either changes — see LANGFUSE_OBSERVABILITY_REFERENCE.md for the known-gap note (NER-only
name catches on the Node side won't be caught here either).
"""

import re

FINANCIAL_COST_REGEX = re.compile(
    r"(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|fee|revenue|cost|payment)\b)",
    re.IGNORECASE,
)

_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and)"
)
SPELLED_OUT_CURRENCY_REGEX = re.compile(
    rf"\b(?:{_NUMBER_WORD}\s+){{1,10}}(?:dollars?|USD|cents?)\b", re.IGNORECASE
)

CORPORATE_EXCLUSIONS = {
    "Adobe", "Inc", "LatentView", "Analytics", "Corporation", "Business", "Head", "Sr", "Manager",
    "Corporate", "Services", "Strategic", "Sourcing", "Director", "Shared", "Operations", "LCM",
    "EMEA", "Support", "Contract", "Delivery", "Consultants", "Work", "Product", "Deliverables",
    "December", "November", "February", "May", "August", "Project", "There", "However", "The", "In",
}

LOCATION_EXCLUSIONS = {
    "Eiffel", "Tower", "Taj", "Mahal", "Golden", "Gate", "Great", "Wall", "Niagara", "Falls",
    "Times", "Square", "Central", "Park", "Statue", "Liberty", "Big", "Ben", "Opera", "House",
    "New", "York", "Delhi", "Las", "Vegas", "Los", "Angeles", "San", "Francisco", "Hong", "Kong",
    "South", "North", "United", "States", "Kingdom", "Sri", "Lanka", "Saudi", "Arabia",
}

COMMON_PERSON_NAMES = {
    "Kumar", "Abhinav", "Rajesh", "Suresh", "Ramesh", "Priya", "Anita", "Vijay", "Arjun",
    "Deepak", "Sanjay", "Ravi", "Ajay", "Vikram", "Nikhil", "Rohit", "Amit", "Sunil", "Manoj",
    "Pankaj", "Sinha", "Sharma", "Gupta", "Verma", "Nair", "Reddy", "Iyer", "Menon", "Pillai", "Rao",
}

PROPER_NOUN_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def _mask_name(match: re.Match) -> str:
    matched_name = match.group(0)
    tokens = matched_name.split()
    if len(tokens) == 1:
        return "[REDACTED PII / NAME Block]" if matched_name in COMMON_PERSON_NAMES else matched_name
    if len(tokens) >= 3:
        return matched_name
    is_corporate = any(t in CORPORATE_EXCLUSIONS for t in tokens)
    is_location = any(t in LOCATION_EXCLUSIONS for t in tokens)
    return matched_name if (is_corporate or is_location) else "[REDACTED PII / NAME Block]"


def mask_financial_and_known_names(text: str) -> str:
    """Redacts financial figures (numeral and spelled-out) and known/common person names."""
    if not text:
        return text
    masked = FINANCIAL_COST_REGEX.sub("[REDACTED COST/REVENUE Metric]", text)
    masked = SPELLED_OUT_CURRENCY_REGEX.sub("[REDACTED COST/REVENUE Metric]", masked)
    masked = PROPER_NOUN_NAME_PATTERN.sub(_mask_name, masked)
    return masked
