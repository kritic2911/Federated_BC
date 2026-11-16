"""
Unit tests for Secure Aggregation (Pairwise Additive Masking) and Minimum-Cohort Guard.
Phase P4 Systems Verification.
"""

import pytest
import torch

from src.systems.secagg import (
    CohortGuardViolationError,
    check_cohort_guard,
    compute_pairwise_mask,
    mask_model_weights,
    unmask_and_aggregate,
)


def test_cohort_guard():
    # Satisfied guard
    assert check_cohort_guard(cohort_size=4, min_cohort_k=3) is True
    assert check_cohort_guard(cohort_size=3, min_cohort_k=3) is True

    # Violated guard
    with pytest.raises(CohortGuardViolationError):
        check_cohort_guard(cohort_size=2, min_cohort_k=3)

    with pytest.raises(CohortGuardViolationError):
        check_cohort_guard(cohort_size=0, min_cohort_k=1)


def test_pairwise_mask_zero_sum_cancellation():
    cohort = ["client_vindr", "client_cmmd", "client_inbreast", "client_cdd"]
    num_params = 1000
    round_seed = 101

    masks = [
        compute_pairwise_mask(client_id, cohort, num_params, round_seed)
        for client_id in cohort
    ]

    # Sum of all masks must be exactly zero (within float32 precision)
    total_mask = sum(masks)
    max_abs_err = torch.max(torch.abs(total_mask)).item()
    assert max_abs_err < 1e-6, f"Mask cancellation failed: max abs error {max_abs_err}"


def test_privacy_guarantee():
    cohort = ["client_vindr", "client_cmmd", "client_inbreast", "client_cdd"]
    round_seed = 42

    dummy_weights = {
        "conv.weight": torch.randn(8, 3, 3, 3),
        "fc.bias": torch.randn(3),
    }

    masked_vec, shapes, dtypes = mask_model_weights(
        weights=dummy_weights,
        client_id="client_vindr",
        cohort=cohort,
        weight_factor=1.0,
        round_seed=round_seed,
    )

    # Flatten raw weights
    from src.systems.secagg import flatten_state_dict
    raw_flat, _, _ = flatten_state_dict(dummy_weights)

    # The masked vector must NOT equal raw weights (protecting client privacy from server)
    diff = torch.norm(masked_vec - raw_flat).item()
    assert diff > 1.0, f"Privacy violation: masked vector is too close to raw weights (diff={diff})"


def test_secagg_unmask_and_aggregate_equivalence():
    cohort = ["client_vindr", "client_cmmd", "client_inbreast", "client_cdd"]
    min_cohort_k = 3
    round_seed = 2026

    # Create distinct local weights for each hospital
    client_weights = {
        c: {
            "conv.weight": torch.randn(4, 3, 3, 3) * (i + 1),
            "linear.bias": torch.randn(3) + i,
        }
        for i, c in enumerate(cohort)
    }

    # Uniform weighting: weight_factor = 1.0 for all
    masked_updates = []
    shapes = None
    dtypes = None
    for c in cohort:
        masked_vec, shapes, dtypes = mask_model_weights(
            weights=client_weights[c],
            client_id=c,
            cohort=cohort,
            weight_factor=1.0,
            round_seed=round_seed,
        )
        masked_updates.append(masked_vec)

    # Aggregate through secure aggregation
    secagg_result = unmask_and_aggregate(
        masked_updates=masked_updates,
        cohort=cohort,
        min_cohort_k=min_cohort_k,
        shapes=shapes,
        dtypes=dtypes,
        total_weight=float(len(cohort)),
    )

    # Compute ground truth unmasked average
    for param_name in ["conv.weight", "linear.bias"]:
        unmasked_avg = sum(client_weights[c][param_name] for c in cohort) / len(cohort)
        assert torch.allclose(secagg_result[param_name], unmasked_avg, atol=1e-5), (
            f"Equivalence failed for {param_name}: SecAgg does not match unmasked mean."
        )

