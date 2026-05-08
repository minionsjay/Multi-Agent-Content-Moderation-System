"""
LangGraph DAG — cold path only.

Routes:
  text_only  → text_agent → decision → action
  image_only → image_agent → decision → action
  mixed      → image_agent → text_agent → decision → action
"""

from langgraph.graph import StateGraph, END
from src.state import ModerationState
from src.agents.text_agent import text_specialist
from src.agents.image_agent import image_specialist
from src.agents.decision import decision_aggregator
from src.agents.action import action_executor


def build_graph() -> StateGraph:
    builder = StateGraph(ModerationState)

    builder.add_node("text_agent", text_specialist)
    builder.add_node("image_agent", image_specialist)
    builder.add_node("decision", decision_aggregator)
    builder.add_node("action", action_executor)

    builder.set_entry_point("image_agent")

    # After image_agent: go to text_agent (OCR text appended), or straight to decision
    builder.add_conditional_edges(
        "image_agent",
        _route_after_image,
        {"text_agent": "text_agent", "decision": "decision"},
    )

    builder.add_edge("text_agent", "decision")
    builder.add_edge("decision", "action")
    builder.add_edge("action", END)

    return builder.compile()


def _route_after_image(state: ModerationState) -> str:
    """After image agent: if text exists (original + OCR), go to text_agent."""
    text = state.get("text", "") or ""
    # If text is non-empty and not just an image URL placeholder
    if text.strip() and not text.strip().startswith("[Image URL:"):
        return "text_agent"
    image_result = state.get("image_result") or {}
    ocr_text = image_result.get("ocr_text", "")
    if ocr_text.strip():
        return "text_agent"
    return "decision"


graph = build_graph()
