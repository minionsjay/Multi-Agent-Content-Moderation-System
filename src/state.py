from typing import TypedDict, Optional, Annotated
from operator import add


class ModerationState(TypedDict, total=False):
    # Original input
    content_id: str
    text: str
    image_url: str
    image_base64: str
    user_id: str
    source: str

    # Triage / Gateway outputs
    content_type: str
    cache_hit: bool
    cached_decision: Optional[dict]
    keyword_confidence: float
    keyword_label: Optional[str]
    keyword_prefiltered: bool   # Gateway already scanned keywords → Text Agent skips L1
    priority_score: float

    # Text Agent outputs
    text_result: Optional[dict]

    # Image Agent outputs
    image_result: Optional[dict]

    # Decision outputs
    decision: str  # pass / block / review
    confidence: float
    reason: str

    # Action outputs
    action_taken: str

    # Model overrides (per-request)
    bert_model: str
    llm_provider: str
    llm_model: str

    # Trace log: each node appends step records, LangGraph reducer concatenates
    traces: Annotated[list[dict], add]
