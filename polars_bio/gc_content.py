import polars_bio as pb
import polars as pl
import pandas as pd
import pyarrow as pa
import tempfile
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union

from polars_bio.context import ctx, set_option
from polars_bio.io import read_fastq
from polars_bio.polars_bio import InputFormat, py_register_table, register_gc_content_udf


def gc_content(
    df: Union[str, pl.LazyFrame, pl.DataFrame, pd.DataFrame],
    output_type: str = "polars.DataFrame",
    target_partitions: int = 4,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Compute GC‐content statistics. Returns a DataFrame with columns:
      - 'gc'    (UInt8): percent GC (0-100),
      - 'count' (Int64): number of reads with that percent GC.

    Args:
        df: Either a file path (FASTQ/Parquet/CSV/BED/VCF) or
            a Polars LazyFrame/DataFrame or pandas DataFrame.
        output_type: "polars.DataFrame" or "pandas.DataFrame"
        target_partitions: how many partitions to set in DataFusion (default: 4)
    """
    try:
        set_option("datafusion.execution.target_partitions", str(target_partitions))
    except Exception:
        pass

    if isinstance(df, str):
        path = Path(df)
        ext_set = set(path.suffixes)
        valid_exts = {".parquet", ".csv", ".bed", ".vcf", ".fastq"}

        if not (ext_set & valid_exts):
            raise ValueError(
                f"Input file must have one of the extensions: {valid_exts}, got: {ext_set}"
            )

        if ".fastq" in ext_set:
            py_register_table(ctx, str(path), "fastq", InputFormat.Fastq, None)
        elif ".parquet" in ext_set:
            py_register_table(ctx, str(path), "fastq", InputFormat.Parquet, None)
        elif ".csv" in ext_set:
            py_register_table(ctx, str(path), "fastq", InputFormat.Csv, None)
        elif ".vcf" in ext_set:
            py_register_table(ctx, str(path), "fastq", InputFormat.Vcf, None)
        elif ".bed" in ext_set:
            py_register_table(ctx, str(path), "fastq", InputFormat.Bed, None)

        register_gc_content_udf(ctx)

        df_lazy = pb.sql(
            """
            SELECT
              gc_percent(sequence) AS gc,
              COUNT(*)               AS count
            FROM fastq
            GROUP BY gc
            ORDER BY gc
            """
        )

    else:
        if isinstance(df, pl.LazyFrame):
            df_polars = df.collect()
            arrow_tbl = df_polars.to_arrow()
        elif isinstance(df, pl.DataFrame):
            arrow_tbl = df.to_arrow()
        elif isinstance(df, pd.DataFrame):
            arrow_tbl = pa.Table.from_pandas(df)
        else:
            raise TypeError(
                "Unsupported dataframe type. Provide: str, polars.LazyFrame, "
                "polars.DataFrame or pandas.DataFrame"
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp_path = tmp.name
        tmp.close()
        pq.write_table(arrow_tbl, tmp_path)

        py_register_table(ctx, tmp_path, "fastq", InputFormat.Parquet, None)
        register_gc_content_udf(ctx)

        df_lazy = pb.sql(
            """
            SELECT
              gc_percent(sequence) AS gc,
              COUNT(*)               AS count
            FROM fastq
            GROUP BY gc
            ORDER BY gc
            """
        )

    df_polars_out = df_lazy.collect()

    if output_type == "polars.DataFrame":
        return df_polars_out
    elif output_type == "pandas.DataFrame":
        return df_polars_out.to_pandas()
    else:
        raise ValueError("output_type must be 'polars.DataFrame' or 'pandas.DataFrame'")


def plot_gc_content(df: pl.DataFrame, ax=None):
    """
    Draws a simple barplot: %GC (x) vs read count (y).
    """
    ax = ax or plt.gca()
    ax.bar(df["gc"], df["count"])
    ax.set_xlabel("% GC")
    ax.set_ylabel("Reads")
    return ax
