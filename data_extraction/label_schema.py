"""Shared label contract for the VeriDelta annotation approaches."""

PROTOCOL_VERSION = "1.0.0"

CATEGORY_CODES = (
    "INTERFACE_DRIFT",
    "SEQUENCE_INVALIDATION",
    "CHECKER_DESYNC",
    "ASSERTION_ORPHANING",
    "RAL_DESYNC",
    "COVERAGE_REGRESSION",
    "OTHER",
)

FAILURE_CATEGORY_CODES = CATEGORY_CODES[:-1]

LABEL_STATUSES = (
    "CLASSIFIED",
    "OTHER",
    "NOT_APPLICABLE",
    "INSUFFICIENT_EVIDENCE",
)

RUN_STATUSES = ("ERROR", "DRY_RUN")
CATEGORY_APPLICABILITY_STATUSES = (
    "in_scope",
    "paired_design_required",
    "not_present_in_checkout",
)


def normalize_label_result(label_status: str, categories: list[str]) -> dict:
    """Normalize model or rule-based output to the shared category contract."""
    if not isinstance(categories, list):
        categories = []
    valid_categories = list(dict.fromkeys(
        category for category in categories if category in CATEGORY_CODES
    ))
    status = label_status.strip().upper() if isinstance(label_status, str) else ""

    if status not in LABEL_STATUSES:
        if valid_categories == ["OTHER"]:
            status = "OTHER"
        elif any(category != "OTHER" for category in valid_categories):
            status = "CLASSIFIED"
        else:
            status = "INSUFFICIENT_EVIDENCE"

    if status == "CLASSIFIED":
        valid_categories = [category for category in valid_categories if category != "OTHER"]
        if not valid_categories:
            status = "INSUFFICIENT_EVIDENCE"
    elif status == "OTHER":
        valid_categories = ["OTHER"]
    else:
        valid_categories = []

    return {"label_status": status, "categories": valid_categories}