"""LangGraph state machine: local privacy check first, then the local or the commercial LLM."""

import os
import re
from typing import Literal, TypedDict

try:
    from langgraph.graph import END, START, StateGraph

    USING_LANGGRAPH = True
except ImportError:  # keeps the routing logic testable without langgraph installed
    USING_LANGGRAPH = False

try:
    import httpx
except ImportError:  # only needed when mock mode is turned off
    httpx = None

Route = Literal["local_llm", "commercial_llm"]

# Patterns for personal, clinical/scheduling or credential data (Japanese and English).
# Lookarounds are used instead of \b because Japanese characters count as word characters.
SENSITIVE_PATTERNS = [
    re.compile(r"(?<!\d)\d{2,4}-\d{2,4}-\d{4}(?!\d)"),  # phone numbers, e.g. 090-1234-5678
    re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"),  # email addresses
    re.compile(r"(患者|カルテ|診療|診察|症状|住所|氏名|生年月日|電話番号|保険証|予約枠|個人情報|機密|パスワード|秘密鍵)"),
    re.compile(
        r"(?<![A-Za-z])(patient|hipaa|ssn|medical record|confidential|password|secret key)s?(?![A-Za-z])",
        re.IGNORECASE,
    ),
]


class OrchestratorState(TypedDict):
    prompt: str
    force_local: bool
    pii_detected: bool
    route: Route
    routing_reason: str
    model_used: str
    answer: str


def initial_state(prompt: str, force_local: bool = False) -> OrchestratorState:
    return {
        "prompt": prompt,
        "force_local": force_local,
        "pii_detected": False,
        "route": "local_llm",
        "routing_reason": "",
        "model_used": "",
        "answer": "",
    }


def contains_sensitive_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def classify_request_node(state: OrchestratorState) -> dict:
    """Deterministic privacy guardrail. Runs locally before any outbound network call."""
    pii_found = contains_sensitive_data(state["prompt"])
    if state.get("force_local"):
        return {
            "pii_detected": pii_found,
            "route": "local_llm",
            "routing_reason": "Caller enforced local-only execution (force_local=True).",
        }
    if pii_found:
        return {
            "pii_detected": True,
            "route": "local_llm",
            "routing_reason": "Sensitive/PII pattern detected -> kept on the local LLM.",
        }
    return {
        "pii_detected": False,
        "route": "commercial_llm",
        "routing_reason": "No sensitive pattern detected -> routed to the commercial cloud LLM.",
    }


def route_decision(state: OrchestratorState) -> Route:
    return state["route"]


def _mock_enabled() -> bool:
    # Fail safe: anything other than an explicit "false"/"0"/"no" keeps mock mode on.
    return os.getenv("MOCK_LLM_RESPONSES", "true").strip().lower() not in {"false", "0", "no"}


def _require_httpx() -> None:
    if httpx is None:
        raise RuntimeError("httpx is required when MOCK_LLM_RESPONSES=false (pip install httpx).")


def call_local_llm_node(state: OrchestratorState) -> dict:
    """Local LLM served by Ollama on the workstation (mocked by default)."""
    model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-coder:14b")
    if _mock_enabled():
        return {
            "model_used": model,
            "answer": f"[MOCK LOCAL:{model}] Processed on-premises; no external network call was made.",
        }
    _require_httpx()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = {"model": model, "prompt": state["prompt"], "stream": False}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {"model_used": model, "answer": data.get("response", "")}


def call_commercial_llm_node(state: OrchestratorState) -> dict:
    """Commercial cloud LLM behind an OpenAI-compatible API (mocked by default)."""
    model = os.getenv("COMMERCIAL_LLM_MODEL", "gpt-4o-mini")
    if _mock_enabled():
        return {
            "model_used": model,
            "answer": f"[MOCK COMMERCIAL:{model}] Non-sensitive prompt handled by the cloud LLM.",
        }
    api_key = os.getenv("COMMERCIAL_LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("COMMERCIAL_LLM_API_KEY is not set (or keep MOCK_LLM_RESPONSES=true).")
    _require_httpx()
    base_url = os.getenv("COMMERCIAL_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": state["prompt"]}]}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {"model_used": model, "answer": data["choices"][0]["message"]["content"]}


class _SequentialFallback:
    """Same invoke() contract as a compiled LangGraph graph, used only if langgraph is missing."""

    def invoke(self, state: OrchestratorState) -> OrchestratorState:
        state = {**state, **classify_request_node(state)}
        if route_decision(state) == "local_llm":
            node = call_local_llm_node
        else:
            node = call_commercial_llm_node
        return {**state, **node(state)}


def build_orchestrator_graph():
    """Compile the LangGraph StateGraph (or the sequential fallback without langgraph)."""
    if not USING_LANGGRAPH:
        return _SequentialFallback()
    builder = StateGraph(OrchestratorState)
    builder.add_node("classify", classify_request_node)
    builder.add_node("local_llm", call_local_llm_node)
    builder.add_node("commercial_llm", call_commercial_llm_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_decision,
        {"local_llm": "local_llm", "commercial_llm": "commercial_llm"},
    )
    builder.add_edge("local_llm", END)
    builder.add_edge("commercial_llm", END)
    return builder.compile()


orchestrator = build_orchestrator_graph()
