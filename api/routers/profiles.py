"""
Profile management endpoints for inspecting, resetting, and deleting user
behavioural profiles.
"""
import logging
import time
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, status
from api.dependencies import (
    get_current_user,
    get_population_prior,
    get_profile_store,
    require_admin,
    rate_limit_check,
)
from api.schemas import (
    ProfileDeleteResponse,
    ProfileResetResponse,
    ProfileResponse,
)
from core.action import UserAction
from core.profile import BehavioralProfile

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/profile",
    tags=["Profiles"],
    responses={
        401: {"description": "Unauthorised"},
        403: {"description": "Forbidden — insufficient permissions"},
        404: {"description": "Profile not found"},
    },
)


@router.get(
    "/{user_id}",
    response_model=ProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get behavioural profile statistics",
    description=(
        "Returns statistical summary of the user's stored behavioural profile. "
        "Does not return raw action logs or resource identifiers. "
        "Users can only access their own profile. Admins can access any profile."
    ),
)
async def get_profile(
    user_id:         str,
    request:         Request,
    current_user_id: Annotated[str, Depends(get_current_user)],
    _rate:           Annotated[None, Depends(rate_limit_check)] = None,
) -> ProfileResponse:
    """
    Return the statistical summary of a user's behavioural profile.
    """
    # Access control: users can only read their own profile
    if current_user_id != user_id:
        roles = getattr(request.state, "roles", [])
        if "admin" not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only access your own profile.",
            )

    profile_store = get_profile_store(request)
    try:
        profile = profile_store.get(user_id)
    except Exception as e:
        logger.error("Profile retrieval failed: user=%s error=%s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile store temporarily unavailable.",
        )

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile found for user '{user_id}'.",
        )

    mean_vec         = profile.get_mean_vector()
    mean_iv, std_iv  = profile.get_timing_stats()
    age_days         = profile.profile_age_seconds() / 86400.0

    action_proportions = {
        action_type: round(mean_vec[i], 4)
        for i, action_type in enumerate(UserAction.ACTION_TYPES)
    }

    return ProfileResponse(
        user_id             = user_id,
        vector_count        = profile.vector_count,
        adaptive_threshold  = round(profile.adaptive_threshold, 4),
        is_mature           = profile.is_mature(),
        training_samples    = len(profile.ml_training_data),
        mean_interval_s     = round(mean_iv, 3),
        std_interval_s      = round(std_iv, 3),
        action_proportions  = action_proportions,
        last_updated        = profile.last_updated,
        profile_age_days    = round(age_days, 2),
    )

@router.delete(
    "/{user_id}",
    response_model=ProfileDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete profile (GDPR right to erasure)",
    description=(
        "Permanently deletes the user's behavioural profile and all associated "
        "data from Redis. This is the technical implementation of GDPR Article 17 "
    ),
)
async def delete_profile(
    user_id:         str,
    request:         Request,
    current_user_id: Annotated[str, Depends(get_current_user)],
) -> ProfileDeleteResponse:
    """
    Delete a user's profile — GDPR right to erasure.
    All stored data is permanently removed:
      - BehavioralProfile (Redis)
      - Consent records (ConsentGatedLogger — if applicable)
    """
    
    if current_user_id != user_id:
        roles = getattr(request.state, "roles", [])
        if "admin" not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete your own profile.",
            )

    profile_store = get_profile_store(request)
    deleted_at    = time.time()
    before_profile = None
    try:
        before_profile = profile_store.get(user_id)
    except Exception:
        before_profile = None

    # Delete from Redis
    deleted = False
    try:
        deleted = profile_store.delete(user_id)
    except Exception as e:
        logger.error("Profile deletion failed: user=%s error=%s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile store temporarily unavailable.",
        )

    # Delete from feature store
    feature_store = getattr(request.app.state, "feature_store", None)
    if feature_store:
        try:
            feature_store.delete_user_records(user_id)
        except Exception as e:
            logger.warning(
                "Feature store deletion failed: user=%s error=%s", user_id, e
            )

    # Clear device history
    device_bridge = getattr(request.app.state, "device_bridge", None)
    if device_bridge:
        try:
            device_bridge.clear_user_history(user_id)
        except Exception as e:
            logger.warning(
                "Device history clear failed: user=%s error=%s", user_id, e
            )

    # Audit trail
    request.state.audit_event = {
        "event_type":  "gdpr_erasure",
        "actor_id":    current_user_id,
        "user_id":     user_id,
        "deleted_by":  current_user_id,
        "deleted_at":  deleted_at,
        "before_state": {
            "profile_exists": bool(before_profile is not None),
            "vector_count": (
                int(before_profile.vector_count)
                if before_profile is not None else 0
            ),
            "training_samples": (
                len(getattr(before_profile, "ml_training_data", []))
                if before_profile is not None else 0
            ),
        },
        "after_state": {
            "profile_exists": False,
            "deleted": bool(deleted),
            "erasure_complete": True,
        },
        "at": deleted_at,
    }

    logger.info(
        "GDPR erasure completed: user=%s deleted_by=%s found=%s",
        user_id, current_user_id, deleted
    )

    return ProfileDeleteResponse(
        user_id          = user_id,
        deleted          = deleted,
        erasure_complete = True,
        deleted_at       = deleted_at,
    )


@router.post(
    "/{user_id}/reset",
    response_model=ProfileResetResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset profile to population prior (admin only)",
    description=(
        "Resets the user's behavioural profile to the population prior baseline, "
        "effectively restarting their cold start period. "
        "This is a security-sensitive operation — admin role required. "
    ),
)
async def reset_profile(
    user_id:    str,
    request:    Request,
    admin_user: Annotated[str, Depends(require_admin)],
) -> ProfileResetResponse:
    """
    Reset a user's profile to the population prior (admin operation).
    """
    profile_store    = get_profile_store(request)
    population_prior = get_population_prior(request)
    before_profile = None

    reset_at = time.time()

    try:
        before_profile = profile_store.get(user_id)
    except Exception:
        before_profile = None

    if population_prior is not None:
        new_profile = population_prior.create_new_user_profile(user_id)
        reset_to    = "population_prior"
    else:
        new_profile = BehavioralProfile(user_id=user_id)
        reset_to    = "empty"

    try:
        profile_store.save(user_id, new_profile)
    except Exception as e:
        logger.error("Profile reset save failed: user=%s error=%s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to save reset profile.",
        )

    # Audit
    request.state.audit_event = {
        "event_type":   "profile_reset",
        "actor_id":     admin_user,
        "user_id":      user_id,
        "reset_to":     reset_to,
        "reset_by":     admin_user,
        "reset_at":     reset_at,
        "before_state": {
            "profile_exists": bool(before_profile is not None),
            "vector_count": (
                int(before_profile.vector_count)
                if before_profile is not None else 0
            ),
            "is_mature": (
                bool(before_profile.is_mature())
                if before_profile is not None else False
            ),
        },
        "after_state": {
            "profile_exists": True,
            "reset_to": reset_to,
            "vector_count": int(new_profile.vector_count),
            "is_mature": bool(new_profile.is_mature()),
        },
        "at": reset_at,
    }

    logger.info(
        "Profile reset: user=%s reset_to=%s by=%s",
        user_id, reset_to, admin_user
    )

    return ProfileResetResponse(
        user_id          = user_id,
        reset_to         = reset_to,
        new_vector_count = new_profile.vector_count,
        reset_at         = reset_at,
    )
