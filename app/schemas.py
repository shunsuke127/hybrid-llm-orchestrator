"""Pydantic v2 request/response contracts for the orchestrator API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Route = Literal["local_llm", "commercial_llm"]


class OrchestrateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User query or scheduling/analysis instruction.")
    force_local: bool = Field(
        default=False,
        description="If true, the prompt is never sent to the cloud route, even if no PII is detected.",
    )
    encrypt_response: bool = Field(
        default=False,
        description=(
            "If true, the answer is returned as AES-256-CBC ciphertext plus its Base64 IV "
            "instead of plaintext."
        ),
    )


class EncryptedPayload(BaseModel):
    ciphertext_b64: str
    iv_b64: str
    algorithm: str = "AES-256-CBC-PKCS7"


class OrchestrateResponse(BaseModel):
    route: Route
    model_used: str
    routing_reason: str
    pii_detected: bool
    answer: Optional[str] = None
    encrypted_payload: Optional[EncryptedPayload] = None
    latency_ms: float
