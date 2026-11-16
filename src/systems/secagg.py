"""
Secure Aggregation (Pairwise Additive Masking) & Minimum-Cohort Guard.
Phase P4: Cross-Silo FL Systems Module.

================================================================================
LEARNING CONTRACT (Rule 10 in AGENTS.md):
Students implement the core mathematical logic for:
  1. compute_pairwise_mask(...)
  2. mask_model_weights(...)
  3. unmask_and_aggregate(...)
  4. check_cohort_guard(...)
================================================================================
"""

import hashlib
import struct
from typing import Dict, List, Tuple

import torch


class CohortGuardViolationError(Exception):
    """Raised when the surviving cohort size is strictly below the minimum guard k."""
    pass


def get_shared_pair_seed(client_u: str, client_v: str, round_seed: int) -> int:
    """
    Deterministically computes a symmetric 32-bit pseudo-random seed shared
    between client_u and client_v for a specific communication round.

    Guarantees:
      get_shared_pair_seed(u, v, r) == get_shared_pair_seed(v, u, r)
    """
    sorted_pair = sorted([client_u, client_v])
    seed_str = f"{sorted_pair[0]}<->{sorted_pair[1]}_round_{round_seed}"
    digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return struct.unpack(">I", digest[:4])[0]


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Size], Dict[str, torch.dtype]]:
    """Flattens a dictionary of tensors into a single contiguous 1D float32 tensor."""
    flat_parts = []
    shapes = {}
    dtypes = {}
    for k, v in state_dict.items():
        shapes[k] = v.shape
        dtypes[k] = v.dtype
        flat_parts.append(v.detach().cpu().to(torch.float32).reshape(-1))
    flat_tensor = torch.cat(flat_parts, dim=0)
    return flat_tensor, shapes, dtypes


def unflatten_state_dict(
    flat_tensor: torch.Tensor,
    shapes: Dict[str, torch.Size],
    dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Reconstructs the original state_dict from a 1D flat tensor using recorded shapes."""
    reconstructed = {}
    offset = 0
    for k, shape in shapes.items():
        num_el = int(torch.prod(torch.tensor(shape)).item()) if len(shape) > 0 else 1
        sub_flat = flat_tensor[offset : offset + num_el]
        reconstructed[k] = sub_flat.reshape(shape).to(dtypes[k])
        offset += num_el
    return reconstructed


# ==============================================================================
# STUDENT TODO IMPLEMENTATIONS BELOW
# ==============================================================================

def check_cohort_guard(cohort_size: int, min_cohort_k: int) -> bool:
    """
    Validates whether the surviving client cohort satisfies the minimum-cohort guard.

    Arguments:
        cohort_size: Number of clients that successfully reached this phase.
        min_cohort_k: Minimum threshold required by config (e.g., configs/config.yaml: systems.min_cohort_k).

    Returns:
        True if cohort_size >= min_cohort_k.

    Raises:
        CohortGuardViolationError: if cohort_size < min_cohort_k.
    """
    if cohort_size < min_cohort_k:
        raise CohortGuardViolationError(f"Cohort size {cohort_size} if below minimum guard threshold k = {min_cohort_k}. Round aborted.")
    return True

def compute_pairwise_mask(
    client_id: str,
    cohort: List[str],
    num_params: int,
    round_seed: int,
) -> torch.Tensor:
    """
    Generates the total additive perturbation mask M_u for client_id across surviving cohort U.

    Mathematical Formulation:
      For each peer v in cohort U (where v != client_id):
        1. Derive symmetric seed: s_uv = get_shared_pair_seed(client_id, v, round_seed)
        2. Generate pseudo-random perturbation vector:
             g = torch.Generator().manual_seed(s_uv)
             p_uv = torch.randn(num_params, generator=g, dtype=torch.float32)
        3. Determine directional sign:
             If client_id < v: sign = +1.0
             If client_id > v: sign = -1.0
        4. Accumulate mask:
             M_u += sign * p_uv

    Key Property:
      Sum_{u in U} M_u == 0 (all pairwise masks identically cancel upon summation).
    """
    total_mask = torch.zeros(num_params, dtype=torch.float32)
    for peer in cohort:
        if peer == client_id:
            continue
        seed = get_shared_pair_seed(client_id, peer, round_seed)
        g = torch.Generator().manual_seed(seed)
        p_uv = torch.randn(num_params, generator=g, dtype=torch.float32)
        sign = 1.0 if client_id < peer else -1.0
        total_mask += sign * p_uv

    return total_mask


def mask_model_weights(
    weights: Dict[str, torch.Tensor],
    client_id: str,
    cohort: List[str],
    weight_factor: float = 1.0,
    round_seed: int = 42,
) -> Tuple[torch.Tensor, Dict[str, torch.Size], Dict[str, torch.dtype]]:
    """
    Flattens and masks a client's local model weights with its pairwise additive mask.

    Formula:
      y_u = (weight_factor * flat_w_u) + M_u

    Returns:
      masked_flat_vector, shapes, dtypes
    """
    flat_weights, shapes, dtypes = flatten_state_dict(weights)
    mask = compute_pairwise_mask(client_id, cohort, len(flat_weights), round_seed)
    masked_vector = (weight_factor * flat_weights) + mask
    return masked_vector, shapes, dtypes


def unmask_and_aggregate(
    masked_updates: List[torch.Tensor],
    cohort: List[str],
    min_cohort_k: int,
    shapes: Dict[str, torch.Size],
    dtypes: Dict[str, torch.dtype],
    total_weight: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregates masked vectors from all surviving clients in cohort U.

    Mathematical Invariant:
      Sum_{u in U} y_u = Sum_{u in U} [ w_u * n_u + M_u ]
                       = Sum_{u in U} [ w_u * n_u ] + Sum_{u in U} M_u
                       = Sum_{u in U} [ w_u * n_u ] + 0

    Steps:
      1. Verify cohort guard: check_cohort_guard(len(masked_updates), min_cohort_k)
      2. Sum all masked_updates element-wise.
      3. If total_weight is provided, normalize: aggregated_flat /= total_weight.
         (If uniform weighting, total_weight = len(cohort)).
      4. Reconstruct dictionary via unflatten_state_dict(...).
    """


    check_cohort_guard(len(masked_updates), min_cohort_k)
    aggregated_flat = sum(masked_updates)
    if total_weight is not None and total_weight > 0:
        aggregated_flat = aggregated_flat / total_weight
    return unflatten_state_dict(aggregated_flat, shapes, dtypes)
