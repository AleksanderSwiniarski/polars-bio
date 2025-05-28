import polars as pl
from polars_bio.io import read_fastq
import matplotlib.pyplot as plt
from datafusion import SessionContext
from polars_bio.polars_bio import register_gc_content_udf   # pyo3-generated

def gc_content(df_or_path, target_partitions=4):
    if isinstance(df_or_path, str):
        df = read_fastq(df_or_path)
    else:
        df = df_or_path

    ctx = SessionContext()
    ctx.session_config = ctx.session_config.replace(
        target_partitions=target_partitions
    )
    ctx.register_dataframe("fastq", df.to_arrow())
    register_gc_content_udf(ctx)

    result = ctx.sql("""
        SELECT gc_percent(seq) AS gc, COUNT(*) AS count
        FROM fastq
        GROUP BY gc
        ORDER BY gc
    """).collect()

    return pl.from_arrow(result)

def plot_gc_content(df, ax=None):
    ax = ax or plt.gca()
    ax.bar(df["gc"], df["count"])
    ax.set_xlabel("% GC")
    ax.set_ylabel("Reads")
    return ax
