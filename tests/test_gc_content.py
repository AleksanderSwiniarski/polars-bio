import os

import pytest
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from polars_bio.io import read_fastq
from polars_bio.gc_content import gc_content, plot_gc_content

EXAMPLE_FASTQ = os.path.join(
    os.path.dirname(__file__), "..", "tests", "data", "example.fastq"
)

class TestGCContent:
    _pd_small = pd.DataFrame({"sequence": ["GC", "AT", "GG", "CC", "TA"]})
    _expected_pd_small = pd.DataFrame({"gc": [0, 100], "count": [2, 3]})
    _expected_pd_small["gc"] = _expected_pd_small["gc"].astype("uint8")
    _expected_pd_small["count"] = _expected_pd_small["count"].astype("int64")

    _pl_small = pl.DataFrame({"sequence": ["GCGC", "ATAT", "CCGG", "TTAA"]})
    _expected_pl_small = pd.DataFrame({"gc": [0, 100], "count": [2, 2]})
    _expected_pl_small["gc"] = _expected_pl_small["gc"].astype("uint8")
    _expected_pl_small["count"] = _expected_pl_small["count"].astype("int64")

    _lf = read_fastq(EXAMPLE_FASTQ)
    _df_gc_pl = gc_content(_lf, output_type="polars.DataFrame")
    _df_gc_pd = gc_content(EXAMPLE_FASTQ, output_type="pandas.DataFrame")

    def test_small_pandas_df(self):
        result_pd = gc_content(self._pd_small, output_type="pandas.DataFrame")
        result_pd = result_pd.sort_values("gc").reset_index(drop=True)
        pd.testing.assert_frame_equal(result_pd, self._expected_pd_small)

        result_pl = gc_content(self._pd_small, output_type="polars.DataFrame")
        result_pl_pd = (
            result_pl.to_pandas()
            .sort_values("gc")
            .reset_index(drop=True)
            .astype({"gc": "uint8", "count": "int64"})
        )
        pd.testing.assert_frame_equal(result_pl_pd, self._expected_pd_small)

    def test_small_polars_df(self):
        result_pl = gc_content(self._pl_small, output_type="polars.DataFrame")
        result_pl_pd = (
            result_pl.to_pandas()
            .sort_values("gc")
            .reset_index(drop=True)
            .astype({"gc": "uint8", "count": "int64"})
        )
        pd.testing.assert_frame_equal(result_pl_pd, self._expected_pl_small)

        result_pd = gc_content(self._pl_small, output_type="pandas.DataFrame")
        result_pd = (
            result_pd.sort_values("gc")
            .reset_index(drop=True)
            .astype({"gc": "uint8", "count": "int64"})
        )
        pd.testing.assert_frame_equal(result_pd, self._expected_pl_small)

    def test_lazyframe_total_counts_and_types(self):
        df_full = self._lf.collect()
        total_reads = df_full.height
        assert self._df_gc_pl["count"].sum() == total_reads
        assert self._df_gc_pl["gc"].dtype == pl.UInt8
        assert self._df_gc_pl["count"].dtype == pl.Int64

    def test_fastq_path_pandas_output(self):
        df_gc_pd = (
            gc_content(EXAMPLE_FASTQ, output_type="pandas.DataFrame")
            .astype({"gc": "uint8", "count": "int64"})
        )
        df_full = self._lf.collect()
        total_reads = df_full.height

        assert int(df_gc_pd["count"].sum()) == total_reads
        assert df_gc_pd["gc"].min() >= 0
        assert df_gc_pd["gc"].max() <= 100
        assert pd.api.types.is_integer_dtype(df_gc_pd["gc"])
        assert pd.api.types.is_integer_dtype(df_gc_pd["count"])

    def test_plot_gc_content_basic(self):
        df_simple = pl.DataFrame({"gc": [0, 50, 100], "count": [5, 10, 5]})
        fig, ax = plt.subplots()
        ax = plot_gc_content(df_simple, ax=ax)

        patches = ax.patches
        assert len(patches) == 3
        heights = [p.get_height() for p in patches]
        assert heights == [5, 10, 5]
        x_positions = [p.get_x() + p.get_width() / 2 for p in patches]
        assert pytest.approx(x_positions, rel=1e-2) == [0, 50, 100]
        plt.close(fig)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            _ = gc_content([1, 2, 3], output_type="polars.DataFrame")

    def test_invalid_output_type(self):
        with pytest.raises(ValueError):
            _ = gc_content(EXAMPLE_FASTQ, output_type="unknown.Format")

    def test_invalid_file_extension(self, tmp_path):
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("ABCD")
        with pytest.raises(ValueError):
            _ = gc_content(str(bad_file), output_type="polars.DataFrame")
