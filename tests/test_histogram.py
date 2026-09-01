"""Log-binned dwell-time histogram arithmetic."""

import math

import numpy as np
import pytest

from dcio.analysis.histogram import (
    bins_per_decade,
    log_bin_edges,
    log_bin_histogram,
    staircase,
)


class TestBinsPerDecade:

    @pytest.mark.parametrize("n,expected", [
        (1, 5), (300, 5), (301, 8), (1000, 8),
        (1001, 10), (3000, 10), (3001, 12), (100000, 12),
    ])
    def test_boundaries(self, n, expected):
        assert bins_per_decade(n) == expected


class TestLogBinEdges:

    def test_ratio_is_constant(self):
        edges, nbdec = log_bin_edges(np.full(500, 0.01), tres=1e-5)
        ratios = edges[1:] / edges[:-1]
        np.testing.assert_allclose(ratios, ratios[0])
        assert ratios[0] == pytest.approx(10.0 ** (1.0 / nbdec))

    def test_nbdec_bins_span_a_decade(self):
        edges, nbdec = log_bin_edges(np.full(500, 0.01), tres=1e-5)
        assert edges[nbdec] / edges[0] == pytest.approx(10.0)

    def test_starts_at_tres(self):
        edges, _ = log_bin_edges(np.full(100, 0.01), tres=2.5e-5)
        assert edges[0] == pytest.approx(2.5e-5)

    def test_nbdec_override_is_honoured(self):
        edges, nbdec = log_bin_edges(np.full(5000, 0.01), tres=1e-5, nbdec=6)
        assert nbdec == 6
        assert edges[6] / edges[0] == pytest.approx(10.0)

    @pytest.mark.parametrize("largest", [5.0, 0.05, 0.5, 30.0, 0.0403, 1.0])
    def test_last_edge_reaches_a_whole_decade(self, largest):
        """The upper limit rounds up to a power of ten, not a power of e.

        Writing that round-up as exp(ceil(log(max))) -- as earlier code in this
        stack did -- gives a power of e, which can fall below the longest
        interval."""
        x = np.array([1e-4, largest])
        edges, _ = log_bin_edges(x, tres=2.5e-5)
        decade = 10.0 ** math.ceil(math.log10(largest))
        assert edges[-1] >= decade

    def test_no_interval_falls_outside_the_bins(self):
        """The regression that motivated moving this down.

        EKDIST's upper limit could land below max(X); numpy.histogram then
        drops those intervals with no warning.  Across randomly drawn
        exponential samples it happened in about 3 of every 10."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(50, 6000))
            x = rng.exponential(0.005, n) + 2.5e-5
            edges, _ = log_bin_edges(x, tres=2.5e-5)
            assert x.max() <= edges[-1]
            assert x.min() >= edges[0]

    def test_rejects_nonpositive_tres(self):
        with pytest.raises(ValueError, match="tres must be positive"):
            log_bin_edges(np.array([0.01]), tres=0.0)

    def test_rejects_empty_sample(self):
        with pytest.raises(ValueError, match="no positive interval"):
            log_bin_edges(np.array([]), tres=1e-5)

    def test_ignores_negative_sentinels(self):
        """impose_resolution marks an unfinished final opening with -1."""
        x = np.array([0.001, 0.002, -1.0])
        edges, _ = log_bin_edges(x, tres=1e-5)
        assert edges[-1] >= 0.002


class TestLogBinHistogram:

    def test_counts_every_interval(self):
        rng = np.random.default_rng(1)
        x = rng.exponential(0.005, 2000) + 2.5e-5
        counts, edges, _ = log_bin_histogram(x, tres=2.5e-5)
        assert counts.sum() == len(x)

    def test_edges_are_one_longer_than_counts(self):
        x = np.full(400, 0.01)
        counts, edges, _ = log_bin_histogram(x, tres=1e-5)
        assert len(edges) == len(counts) + 1

    def test_all_equal_intervals_land_in_one_bin(self):
        x = np.full(400, 0.01)
        counts, _, _ = log_bin_histogram(x, tres=1e-5)
        assert (counts > 0).sum() == 1
        assert counts.max() == 400


class TestStaircase:

    def test_vertex_count(self):
        counts = np.array([1, 2, 3])
        edges = np.array([1.0, 2.0, 3.0, 4.0])
        x, y = staircase(edges, counts)
        assert len(x) == len(y) == 2 * len(edges)

    def test_closes_to_zero_at_both_ends(self):
        x, y = staircase(np.array([1.0, 2.0, 3.0]), np.array([4, 5]))
        assert y[0] == 0
        assert y[-1] == 0

    def test_each_bar_is_flat_across_its_bin(self):
        counts = np.array([7, 3])
        edges = np.array([1.0, 2.0, 4.0])
        x, y = staircase(edges, counts)
        # vertices: (1,0) (1,7) (2,7) (2,3) (4,3) (4,0)
        assert list(y) == [0, 7, 7, 3, 3, 0]
        assert list(x) == [1.0, 1.0, 2.0, 2.0, 4.0, 4.0]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="expected 3 edges"):
            staircase(np.array([1.0, 2.0]), np.array([1, 2]))
