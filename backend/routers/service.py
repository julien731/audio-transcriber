from __future__ import annotations

from fastapi import APIRouter

import config
from backend.schemas import HealthResponse, ProvisioningStatus, TokenUpdate
from backend.services import provisioning

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Readiness signal a client polls before presenting UI (BR-5, EC-2).

    Once this endpoint responds the service is accepting requests; the genuine
    pre-HTTP not-ready window is signalled out-of-band via the startup handshake
    / port file (see service_main.py).
    """
    return HealthResponse(
        status="ready",
        provisioning_completed=provisioning.models_ready(),
        diarization_available=bool(config.current_hf_token()),
    )


@router.get("/provisioning", response_model=ProvisioningStatus)
async def get_provisioning():
    return provisioning.status()


@router.post("/provisioning/token", response_model=ProvisioningStatus)
async def set_provisioning_token(payload: TokenUpdate):
    return provisioning.set_token(payload.hf_token)


@router.post("/provisioning/models", response_model=ProvisioningStatus)
async def start_provisioning_models():
    return provisioning.start_download()
