"""Unit tests for privacy_utils.py — RDP accounting, sensitivity tracking, noise allocation."""

import math
import numpy as np
import pytest
import torch

from qpriviot_fl.privacy_utils import (
    RenyiPrivacyAccountant,
    SensitivityTracker,
    allocate_adaptive_noise,
    apply_dp_noise_per_layer,
    quantize,
    dequantize,
    generate_zero_sum_masks,
    skellam_noise,
    skellam_rdp_epsilon,
    apply_distributed_skellam_noise,
)


def _gaussian_rdp_eps(sigma, delta=1e-5, releases=1, orders=range(2, 257)):
    """Gaussian-mechanism RDP ε (leading term only) — the limit the Skellam ε converges to
    as the quantiser resolution grows. min_α releases·α/(2σ²) + ln(1/δ)/(α−1)."""
    return min(releases * a / (2 * sigma * sigma) + math.log(1.0 / delta) / (a - 1.0)
               for a in orders)


# ─── RenyiPrivacyAccountant ───────────────────────────────────────────────────

class TestRenyiPrivacyAccountant:
    def test_zero_rounds_returns_zero(self):
        acc = RenyiPrivacyAccountant(target_epsilon=3.0)
        assert acc.get_total_epsilon() == 0.0

    def test_epsilon_increases_with_rounds(self):
        acc = RenyiPrivacyAccountant(target_epsilon=10.0)
        eps_prev = 0.0
        for _ in range(5):
            acc.add_round(noise_multiplier=1.5, sampling_rate=0.1, steps=1)
            eps_now = acc.get_total_epsilon()
            assert eps_now > eps_prev
            eps_prev = eps_now

    def test_higher_noise_lower_epsilon(self):
        """Higher noise multiplier ⟹ lower privacy cost per round."""
        acc_noisy = RenyiPrivacyAccountant()
        acc_quiet = RenyiPrivacyAccountant()
        acc_noisy.add_round(noise_multiplier=10.0, sampling_rate=0.1, steps=1)
        acc_quiet.add_round(noise_multiplier=1.0, sampling_rate=0.1, steps=1)
        assert acc_noisy.get_total_epsilon() < acc_quiet.get_total_epsilon()

    def test_higher_sampling_rate_higher_epsilon(self):
        """Larger sampling rate ⟹ more privacy cost."""
        acc_high = RenyiPrivacyAccountant()
        acc_low = RenyiPrivacyAccountant()
        acc_high.add_round(noise_multiplier=2.0, sampling_rate=1.0, steps=1)
        acc_low.add_round(noise_multiplier=2.0, sampling_rate=0.1, steps=1)
        assert acc_high.get_total_epsilon() > acc_low.get_total_epsilon()

    def test_remaining_budget_decreases(self):
        acc = RenyiPrivacyAccountant(target_epsilon=5.0)
        remaining_prev = acc.get_remaining_budget()
        assert remaining_prev == 5.0
        for _ in range(3):
            acc.add_round(noise_multiplier=2.0, sampling_rate=0.5, steps=1)
            remaining_now = acc.get_remaining_budget()
            assert remaining_now < remaining_prev
            remaining_prev = remaining_now

    def test_is_budget_exceeded(self):
        acc = RenyiPrivacyAccountant(target_epsilon=0.5)
        assert not acc.is_budget_exceeded()
        # Add many rounds with very low noise to force budget exhaustion
        for _ in range(100):
            acc.add_round(noise_multiplier=0.5, sampling_rate=1.0, steps=1)
        assert acc.is_budget_exceeded()

    def test_zero_sigma_returns_inf(self):
        acc = RenyiPrivacyAccountant()
        acc.add_round(noise_multiplier=0.0, sampling_rate=0.1, steps=1)
        assert acc.get_total_epsilon() == float("inf")

    def test_composition_is_additive_in_rdp(self):
        """Two calls with T steps each == one call with 2T steps."""
        acc_split = RenyiPrivacyAccountant()
        acc_split.add_round(noise_multiplier=3.0, sampling_rate=0.5, steps=1)
        acc_split.add_round(noise_multiplier=3.0, sampling_rate=0.5, steps=1)

        acc_combined = RenyiPrivacyAccountant()
        acc_combined.add_round(noise_multiplier=3.0, sampling_rate=0.5, steps=2)

        assert abs(acc_split.get_total_epsilon() - acc_combined.get_total_epsilon()) < 1e-9


# ─── SensitivityTracker ───────────────────────────────────────────────────────

class TestSensitivityTracker:
    def _make_grads(self, keys, value=0.1):
        return {k: torch.full((4,), value) for k in keys}

    def test_empty_tracker_returns_empty(self):
        tracker = SensitivityTracker()
        assert tracker.get_sensitivities() == {}

    def test_mean_is_one_after_normalization(self):
        tracker = SensitivityTracker()
        keys = ["a", "b", "c", "d"]
        norms = {"a": 1.0, "b": 2.0, "c": 0.5, "d": 3.0}
        grads = self._make_grads(keys)
        # Several rounds so std is non-zero
        for v in [1.0, 2.0, 0.5, 3.0]:
            norms = {"a": v, "b": v * 2, "c": v * 0.5, "d": v * 3}
            tracker.update(norms, grads)
        s = tracker.get_sensitivities()
        assert set(s.keys()) == set(keys)
        mean_s = np.mean(list(s.values()))
        assert abs(mean_s - 1.0) < 1e-6, f"mean={mean_s}, expected 1.0"

    def test_all_sensitivities_positive(self):
        tracker = SensitivityTracker()
        for _ in range(3):
            tracker.update({"x": 0.1, "y": 5.0}, self._make_grads(["x", "y"]))
        for name, s in tracker.get_sensitivities().items():
            assert s > 0, f"sensitivity[{name}]={s} is not positive"

    def test_window_size_respected(self):
        tracker = SensitivityTracker(window_size=3)
        for i in range(10):
            tracker.update({"a": float(i)}, {})
        # History should contain at most 3 entries
        assert len(tracker.history["a"]) <= 3

    def test_higher_variance_yields_higher_sensitivity(self):
        """Layer with more volatile norms should have higher sensitivity."""
        tracker = SensitivityTracker(alpha=0.0)  # pure norm-std component
        for v in [1.0, 5.0, 0.1, 4.0, 0.2]:
            tracker.update({"stable": 1.0, "volatile": v}, {})
        s = tracker.get_sensitivities()
        assert s["volatile"] > s["stable"]


# ─── allocate_adaptive_noise ──────────────────────────────────────────────────

class TestAllocateAdaptiveNoise:
    def _sensitivities(self):
        return {"layer_a": 2.0, "layer_b": 0.5, "layer_c": 1.0}

    def test_returns_correct_keys(self):
        sens = self._sensitivities()
        nm, cn, _ = allocate_adaptive_noise(sens, target_epsilon=3.0)
        assert set(nm.keys()) == set(sens.keys())
        assert set(cn.keys()) == set(sens.keys())

    def test_higher_sensitivity_higher_noise(self):
        sens = self._sensitivities()
        nm, _, _ = allocate_adaptive_noise(sens, target_epsilon=3.0)
        assert nm["layer_a"] > nm["layer_b"], "High sensitivity should get more noise"

    def test_clip_norm_is_constant_across_layers(self):
        """Current design (privacy_utils.py): the adaptivity is a *noise reallocation* at a
        FIXED clip. A constant clip keeps the per-layer sensitivity Δ2 uniform so the
        renormalised noise mean equals base_sigma exactly (iso-privacy); the clip is not
        itself made sensitivity-dependent."""
        sens = self._sensitivities()
        _, cn, _ = allocate_adaptive_noise(sens, target_epsilon=3.0)
        assert len(set(cn.values())) == 1, "clip norm should be constant across layers"

    def test_empty_sensitivities_returns_empty_dicts(self):
        nm, cn, base_sigma = allocate_adaptive_noise({}, target_epsilon=3.0)
        assert nm == {}
        assert cn == {}
        assert base_sigma > 0

    def test_base_sigma_override(self):
        """Explicitly provided base_sigma must be used, not computed from epsilon."""
        sens = self._sensitivities()
        _, _, sigma_default = allocate_adaptive_noise(sens, target_epsilon=3.0)
        _, _, sigma_override = allocate_adaptive_noise(sens, target_epsilon=3.0, base_sigma=99.0)
        # The override sigma should propagate as the base (noise_mults scale it by s_j)
        nm_override, _, _ = allocate_adaptive_noise(sens, target_epsilon=3.0, base_sigma=99.0)
        nm_default, _, _ = allocate_adaptive_noise(sens, target_epsilon=3.0)
        # All override values should be larger (proportionally) than defaults
        ratio = nm_override["layer_c"] / nm_default["layer_c"]
        assert abs(ratio - 99.0 / sigma_default) < 0.1


# ─── quantize / dequantize ────────────────────────────────────────────────────

class TestQuantizeDequantize:
    def test_roundtrip_accuracy(self):
        rng = np.random.default_rng(0)
        params = [rng.uniform(-0.5, 0.5, (10, 10)).astype(np.float32) for _ in range(3)]
        q = quantize(params, clip_range=1.0, range_max=1_000_000)
        recovered = dequantize(q, clip_range=1.0, range_max=1_000_000)
        for orig, rec in zip(params, recovered):
            np.testing.assert_allclose(orig, rec, atol=1e-5)

    def test_values_outside_clip_are_clamped(self):
        large = [np.array([[-5.0, 5.0, 0.0]])]
        q = quantize(large, clip_range=1.0)
        rec = dequantize(q, clip_range=1.0)
        assert rec[0].max() <= 1.0 + 1e-5
        assert rec[0].min() >= -1.0 - 1e-5

    def test_quantized_values_are_integers(self):
        params = [np.array([[0.1, -0.2, 0.3]])]
        q = quantize(params, clip_range=1.0)
        assert q[0].dtype == np.int64


# ─── generate_zero_sum_masks ──────────────────────────────────────────────────

class TestZeroSumMasks:
    def test_masks_sum_to_zero(self):
        shapes = [(10,), (5, 5), (3, 3, 3)]
        masks = generate_zero_sum_masks(shapes, num_clients=4, seed=42)
        for layer_idx, shape in enumerate(shapes):
            total = sum(masks[c][layer_idx] for c in range(4))
            np.testing.assert_array_equal(total, np.zeros(shape, dtype=np.int64))

    def test_correct_number_of_clients(self):
        masks = generate_zero_sum_masks([(8,)], num_clients=5, seed=0)
        assert len(masks) == 5

    def test_deterministic_with_same_seed(self):
        m1 = generate_zero_sum_masks([(4,)], num_clients=3, seed=99)
        m2 = generate_zero_sum_masks([(4,)], num_clients=3, seed=99)
        np.testing.assert_array_equal(m1[0][0], m2[0][0])

    def test_different_with_different_seed(self):
        m1 = generate_zero_sum_masks([(100,)], num_clients=3, seed=1)
        m2 = generate_zero_sum_masks([(100,)], num_clients=3, seed=2)
        assert not np.array_equal(m1[0][0], m2[0][0])


# ─── skellam_noise (§5.5 discrete mechanism + SecAgg closure) ──────────────────

class TestSkellamNoise:
    def test_variance_matches_target(self):
        """Skellam(μ)=Poisson(μ/2)−Poisson(μ/2) has variance = 2·(μ/2) = target."""
        rng = np.random.default_rng(0)
        target = 40.0
        noise = skellam_noise((400_000,), target, rng)
        assert abs(noise.var() - target) / target < 0.05
        assert abs(float(noise.mean())) < 0.1  # symmetric ⇒ zero mean

    def test_is_integer_valued(self):
        rng = np.random.default_rng(0)
        noise = skellam_noise((1000,), 10.0, rng)
        assert noise.dtype == np.int64

    def test_zero_variance_returns_zeros(self):
        rng = np.random.default_rng(0)
        noise = skellam_noise((100,), 0.0, rng)
        np.testing.assert_array_equal(noise, np.zeros(100, dtype=np.int64))

    def test_closure_under_addition(self):
        """§5.5: N clients each add a Skellam share of variance μ/N; because the Skellam
        is closed under addition, the masked SecAgg sum is a single Skellam of variance μ.
        The aggregate variance must equal the target μ — the property the distributed-DP
        guarantee rests on."""
        rng = np.random.default_rng(1)
        mu, N, shape = 40.0, 16, (400_000,)
        agg = np.zeros(shape, dtype=np.int64)
        for _ in range(N):
            agg += skellam_noise(shape, mu / N, rng)
        assert abs(agg.var() - mu) / mu < 0.05

    def test_distributed_aggregate_hits_target_variance(self):
        """apply_distributed_skellam_noise(distributed=True): per-client variance base/N,
        SecAgg sum ⇒ aggregate variance == base_var = (σ·C·s)² (central-DP-equivalent)."""
        N = 16
        empty = [np.zeros((300_000,), dtype=np.int64) for _ in range(N)]
        noised = apply_distributed_skellam_noise(
            empty, sigma=0.606, clip_norm=1.0, num_clients=N,
            quantization_bound=10.0, range_max=1_000_000, distributed=True, seed=1)
        agg = sum(noised)
        base_var = (0.606 * 1.0 * (1_000_000 / 10.0)) ** 2
        assert abs(agg.var() - base_var) / base_var < 0.02

    def test_distributed_uses_less_aggregate_noise_than_local(self):
        """distributed=True (SecAgg, agg var = base) vs distributed=False (local DP, agg
        var = N·base): the distributed path carries ~N× less aggregate noise."""
        N = 16
        empty = [np.zeros((300_000,), dtype=np.int64) for _ in range(N)]
        kw = dict(sigma=0.606, clip_norm=1.0, num_clients=N,
                  quantization_bound=10.0, range_max=1_000_000, seed=2)
        dist = sum(apply_distributed_skellam_noise(empty, distributed=True, **kw))
        local = sum(apply_distributed_skellam_noise(empty, distributed=False, **kw))
        assert local.var() / dist.var() > 0.7 * N  # ≈ N, allow sampling slack


# ─── skellam_rdp_epsilon (§7.5 accounting — "discretisation is free") ──────────

class TestSkellamAccounting:
    SCALE_1E6 = 1_000_000 / 10.0  # range_max=1e6, quantization_bound=10

    def test_zero_sigma_returns_inf(self):
        eps, _ = skellam_rdp_epsilon(0.0, 1.0, self.SCALE_1E6, 1024)
        assert eps == float("inf")

    def test_never_below_gaussian(self):
        """The discretisation surcharge is non-negative: Skellam ε ≥ Gaussian-RDP ε."""
        for sigma in (0.303, 0.606, 1.615):
            eps, _ = skellam_rdp_epsilon(sigma, 1.0, self.SCALE_1E6, 1024, releases=1)
            assert eps >= _gaussian_rdp_eps(sigma) - 1e-9

    def test_discretisation_is_free_at_1e6(self):
        """§7.5: at range_max=1e6 the discrete Skellam ε equals the Gaussian-RDP ε to 4+
        decimals — the integer/SecAgg quantisation costs no privacy."""
        for sigma in (0.303, 0.606, 1.615):
            eps, _ = skellam_rdp_epsilon(sigma, 1.0, self.SCALE_1E6, 1024, releases=1)
            assert abs(eps - _gaussian_rdp_eps(sigma)) < 1e-3

    def test_surcharge_shrinks_with_resolution(self):
        """Coarse quantisation ⇒ real surcharge; ε decreases monotonically toward the
        Gaussian limit as range_max (resolution) grows."""
        eps_prev = None
        for range_max in (1e2, 1e3, 1e4, 1e5, 1e6):
            eps, _ = skellam_rdp_epsilon(0.606, 1.0, range_max / 10.0, 1024, releases=1)
            if eps_prev is not None:
                assert eps <= eps_prev + 1e-9
            eps_prev = eps
        # coarsest resolution carries a visible surcharge over Gaussian
        eps_coarse, _ = skellam_rdp_epsilon(0.606, 1.0, 1e2 / 10.0, 1024, releases=1)
        assert eps_coarse - _gaussian_rdp_eps(0.606) > 1e-2

    def test_dim_independent_at_high_resolution(self):
        """At range_max=1e6 the surcharge (∝ √dim/μ) is negligible, so ε is effectively
        independent of the release dimension — why the ε-relabelling is a single number."""
        e_small, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 32 * 32, releases=1)
        e_large, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 2048 * 32, releases=1)
        assert abs(e_small - e_large) < 1e-4

    def test_higher_sigma_lower_eps(self):
        e_lo, _ = skellam_rdp_epsilon(1.615, 1.0, self.SCALE_1E6, 1024)
        e_hi, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 1024)
        assert e_lo < e_hi

    def test_composition_raises_eps(self):
        """Two identical releases (sum-vector + counts) compose to a larger ε than one."""
        e1, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 1024, releases=1)
        e2, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 1024, releases=2)
        assert e2 > e1

    def test_paper_relabelling_headline(self):
        """The paper's ε-relabelling: the exploration σ's labelled ε=8 / ε=3 via the loose
        classic bound carry rigorous Skellam-RDP ε ≈ 9.3 / 3.2 (vector-only, range_max=1e6)."""
        e8, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 1024, releases=1)
        e3, _ = skellam_rdp_epsilon(1.615, 1.0, self.SCALE_1E6, 1024, releases=1)
        assert abs(e8 - 9.3) < 0.1
        assert abs(e3 - 3.2) < 0.1
        # two-channel design costs ~1.5× the ε of the vector-only release (§7.5)
        e8_2, _ = skellam_rdp_epsilon(0.606, 1.0, self.SCALE_1E6, 1024, releases=2)
        assert abs(e8_2 - 13.9) < 0.2