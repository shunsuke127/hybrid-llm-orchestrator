"""FastAPI transport layer: GET /health and POST /v1/orchestrate."""

import logging
import time

from fastapi import FastAPI, HTTPException

from app.crypto import EncryptionNotConfiguredError, encrypt_aes256_cbc, get_aes_key
from app.graph import USING_LANGGRAPH, initial_state, orchestrator
from app.schemas import EncryptedPayload, OrchestrateRequest, OrchestrateResponse

logger = logging.getLogger("hybrid_llm_orchestrator")

app = FastAPI(
    title="Hybrid Local/Cloud LLM Orchestrator",
    version="1.0.0",
    description=(
        "Privacy-first FastAPI + LangGraph backend that keeps sensitive prompts on a local LLM "
        "(Ollama) and sends non-sensitive prompts to a commercial cloud LLM."
    ),
)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "graph_engine": "langgraph" if USING_LANGGRAPH else "sequential-fallback",
    }


@app.post("/v1/orchestrate", response_model=OrchestrateResponse)
def orchestrate(req: OrchestrateRequest) -> OrchestrateResponse:
    t0 = time.perf_counter()

    key = None
    if req.encrypt_response:
        try:
            key = get_aes_key()  # checked before any LLM call: fail closed, not after the fact
        except EncryptionNotConfiguredError as exc:
            logger.error("encrypt_response=true, but AES_SHARED_SECRET is not set")
            raise HTTPException(
                status_code=500, detail="Response encryption is not configured on the server."
            ) from exc

    try:
        result = orchestrator.invoke(initial_state(req.prompt, req.force_local))
    except Exception as exc:
        logger.exception("LLM invocation failed")
        raise HTTPException(status_code=502, detail="Upstream LLM call failed.") from exc

    answer = result["answer"]
    encrypted_payload = None
    if key is not None:
        ciphertext_b64, iv_b64 = encrypt_aes256_cbc(answer, key)
        encrypted_payload = EncryptedPayload(ciphertext_b64=ciphertext_b64, iv_b64=iv_b64)
        answer = None  # never return plaintext when encryption was requested

    return OrchestrateResponse(
        route=result["route"],
        model_used=result["model_used"],
        routing_reason=result["routing_reason"],
        pii_detected=result["pii_detected"],
        answer=answer,
        encrypted_payload=encrypted_payload,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
