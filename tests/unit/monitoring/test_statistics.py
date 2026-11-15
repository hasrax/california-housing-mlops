import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from housing_mlops.monitoring.statistics import (
    ks_against_reference,
    observed_fractions,
    population_stability_index,
    psi_from_reference,
    reference_cdf,
)
from housing_mlops.training.reference import QUANTILE_GRID, feature_reference

EDGES = [0.0, 1.0, 2.0, 3.0, 4.0]


def test_observed_fractions_sum_to_one_and_open_outer_bins():
    fractions = observed_fractions([-10.0, 0.5, 1.5, 2.5, 3.5, 99.0], EDGES)
    assert fractions.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(fractions, [2 / 6, 1 / 6, 1 / 6, 2 / 6])


@pytest.mark.parametrize("edges", [[1.0], [1.0, 1.0], [2.0, 1.0], []])
def test_observed_fractions_rejects_bad_edges(edges):
    with pytest.raises(ValueError):
        observed_fractions([1.0], edges)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf, 1.0]])
def test_observed_fractions_rejects_bad_values(values):
    with pytest.raises(ValueError):
        observed_fractions(values, EDGES)


def test_psi_is_zero_for_identical_distributions():
    assert population_stability_index([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_known_value():
    expected = 0.4 * math.log(0.9 / 0.5) + (-0.4) * math.log(0.1 / 0.5)
    assert population_stability_index([0.5, 0.5], [0.9, 0.1]) == pytest.approx(expected)


def test_psi_handles_empty_observed_bins_without_infinities():
    value = population_stability_index([0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0])
    assert math.isfinite(value)
    assert value > 1.0


def test_psi_rejects_mismatched_or_negative_input():
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.5, -0.5])


def test_psi_from_reference_is_small_for_same_data_and_grows_with_shift():
    rng = np.random.default_rng(0)
    training = rng.normal(0.0, 1.0, 20_000)
    reference = feature_reference(training)
    same = rng.normal(0.0, 1.0, 2_000)
    shifted = rng.normal(1.0, 1.0, 2_000)
    psi_same = psi_from_reference(reference["bin_fractions"], reference["bin_edges"], same)
    psi_shifted = psi_from_reference(reference["bin_fractions"], reference["bin_edges"], shifted)
    assert psi_same < 0.05
    assert psi_shifted > 0.5
    assert psi_shifted > psi_same


def test_reference_cdf_interpolates_between_quantiles():
    cdf = reference_cdf([0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        cdf([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]), [0, 0, 0.25, 0.5, 0.75, 1, 1]
    )


def test_reference_cdf_treats_duplicate_quantiles_as_a_step():
    cdf = reference_cdf([0.0, 1.0, 1.0, 1.0, 2.0], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert cdf(1.0) == pytest.approx(0.75)
    assert cdf(0.999) < 0.25
    assert cdf(1.5) == pytest.approx(0.875)


def test_reference_cdf_rejects_decreasing_or_mismatched_input():
    with pytest.raises(ValueError):
        reference_cdf([2.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        reference_cdf([0.0, 1.0], [0.0, 0.5, 1.0])


def test_ks_detects_shift_and_accepts_same_distribution():
    rng = np.random.default_rng(1)
    training = rng.normal(0.0, 1.0, 20_000)
    reference = feature_reference(training)
    same = ks_against_reference(reference["quantiles"], QUANTILE_GRID, rng.normal(0, 1, 2_000))
    shifted = ks_against_reference(
        reference["quantiles"], QUANTILE_GRID, rng.normal(2.0, 1.0, 2_000)
    )
    assert same.statistic < 0.05
    assert same.p_value > 0.01
    assert shifted.statistic > 0.5
    assert shifted.p_value < 1e-10


fractions = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=20
)
finite_values = npst.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=200),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@given(fractions)
def test_property_psi_of_a_distribution_with_itself_is_zero(reference):
    assert population_stability_index(reference, reference) == pytest.approx(0.0, abs=1e-12)


@given(
    st.integers(min_value=2, max_value=20).flatmap(
        lambda n: st.tuples(
            st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=n, max_size=n),
            st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=n, max_size=n),
        )
    )
)
def test_property_psi_is_non_negative_and_symmetric(pair):
    reference, observed = pair
    forward = population_stability_index(reference, observed)
    backward = population_stability_index(observed, reference)
    assert forward >= -1e-12
    assert forward == pytest.approx(backward, abs=1e-9)


@given(finite_values)
@settings(max_examples=200)
def test_property_observed_fractions_form_a_distribution(values):
    fractions = observed_fractions(values, EDGES)
    assert fractions.shape == (len(EDGES) - 1,)
    assert np.all(fractions >= 0)
    assert fractions.sum() == pytest.approx(1.0)


@given(finite_values, st.permutations(range(4)))
def test_property_psi_ignores_order_and_duplication_of_the_sample(values, _unused):
    reference = [0.1, 0.2, 0.3, 0.4]
    shuffled = np.random.default_rng(0).permutation(values)
    doubled = np.concatenate([values, values])
    base = psi_from_reference(reference, EDGES, values)
    assert psi_from_reference(reference, EDGES, shuffled) == pytest.approx(base)
    assert psi_from_reference(reference, EDGES, doubled) == pytest.approx(base)


sorted_quantiles = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=2, max_size=50
).map(sorted)


@given(sorted_quantiles, finite_values)
@settings(max_examples=200)
def test_property_ks_statistic_and_p_value_are_probabilities(quantiles, values):
    grid = np.linspace(0.0, 1.0, len(quantiles))
    result = ks_against_reference(quantiles, grid, values)
    assert 0.0 <= result.statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


@given(sorted_quantiles, finite_values)
def test_property_reference_cdf_is_monotonic_and_bounded(quantiles, points):
    grid = np.linspace(0.0, 1.0, len(quantiles))
    cdf = reference_cdf(quantiles, grid)
    evaluated = cdf(np.sort(points))
    assert np.all(np.diff(evaluated) >= -1e-12)
    assert np.all((evaluated >= 0.0) & (evaluated <= 1.0))
