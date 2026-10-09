"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    transform = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    position = (
        transform[:3, :3] @ np.asarray(meas.z, dtype=float).reshape(-1)[:3]
        + transform[:3, 3]
    )
    x = np.asmatrix(np.r_[position, [0.0, 0.0, 0.0]]).T

    rotation = transform[:3, :3]
    position_covariance = rotation @ np.asarray(meas.R) @ rotation.T
    velocity_covariance = np.diag([
        params.sigma_p44**2,
        params.sigma_p55**2,
        params.sigma_p66**2,
    ])
    covariance = np.zeros((params.dim_state, params.dim_state), dtype=float)
    covariance[:3, :3] = position_covariance
    covariance[3:, 3:] = velocity_covariance
    return {
        "x": x,
        "P": np.asmatrix(covariance),
        "score": 1 / params.window,
        "state": "initialized",
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    delta = 1 / params.window
    score = float(track["score"])
    score = min(1.0, score + delta) if associated else score - delta
    track["score"] = score

    if track["state"] == "confirmed":
        return track
    if score > params.confirmed_threshold:
        track["state"] = "confirmed"
    elif associated:
        track["state"] = "tentative"
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    covariance = np.asarray(track["P"], dtype=float)
    if covariance[0, 0] > params.max_P or covariance[1, 1] > params.max_P:
        return True
    if track["state"] == "confirmed":
        return float(track["score"]) < params.delete_threshold
    return float(track["score"]) <= 0
