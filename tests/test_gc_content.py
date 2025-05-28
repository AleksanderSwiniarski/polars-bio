# python/polars_bio/tests/test_gc_content.py
#
# Jednostkowe testy funkcji qc.gc_content.gc_content  (GC-percent UDF + wrapper)
# uruchamiane poleceniem:  pytest -q
#
# Wymagania:
#   pytest
#   polars>=0.20
#   polars-bio (z zaimplementowaną funkcją gc_content)
# --------------------------------------------------------------------------------

import pathlib
import numpy as np
import polars as pl
import pytest

from polars_bio.io import read_fastq
from polars_bio.gc_content import gc_content


# ────────────────────────────────────────────────────────────────────────────────
# Pomocnicza referencyjna implementacja GC-percent w czystym Pythonie
# ────────────────────────────────────────────────────────────────────────────────
def _py_gc_percent(seq: str) -> int:
    """
    Oblicza %GC (0-100) dla pojedynczej sekwencji FASTQ.
    """
    gc = sum(1 for b in seq if b in "GCgc")
    return int(gc * 100 / len(seq)) if seq else 0


# Ścieżki plików testowych zaczerpnięte z upstream fastqc-rs
RES_DIR = pathlib.Path(__file__).with_suffix("").parent / "resources"
FASTQ_PATH = RES_DIR / "example.fastq"


# ────────────────────────────────────────────────────────────────────────────────
# 1. Walidacja wyniku na podstawie referencyjnego obliczenia w Pythonie
# ────────────────────────────────────────────────────────────────────────────────
def test_gc_content_fastq_matches_python_reference():
    """
    Porównujemy wynik gc_content(path) z dystrybucją GC policzoną „ręcznie”.
    """
    # Wynik funkcji z biblioteki
    lib_df = gc_content(str(FASTQ_PATH)).sort("gc")

    # Referencja (czysty Python → Polars → agregacja)
    seq_df = read_fastq(str(FASTQ_PATH))  # kolumna "seq"
    ref_df = (
        seq_df.select(
            pl.col("seq")
            .map_elements(_py_gc_percent, return_dtype=pl.UInt8)
            .alias("gc")
        )
        .group_by("gc")
        .count()
        .sort("gc")
    )

    # Obie ramki muszą mieć identyczne wiersze i kolejność
    assert lib_df.frame_equal(
        ref_df, null_equal=True
    ), "Rozkład %GC różni się od referencji"


# ────────────────────────────────────────────────────────────────────────────────
# 2. Obsługa wejścia jako gotowego DataFrame
# ────────────────────────────────────────────────────────────────────────────────
def test_gc_content_accepts_dataframe_input():
    """
    Funkcja powinna akceptować również `polars.DataFrame`, nie tylko ścieżkę.
    """
    df = read_fastq(str(FASTQ_PATH))
    out = gc_content(df)
    assert isinstance(out, pl.DataFrame)
    # suma liczby odczytów powinna się zgadzać
    assert out["count"].sum() == len(df)


# ────────────────────────────────────────────────────────────────────────────────
# 3. Tolerancja wartości NULL / brakujących sekwencji
# ────────────────────────────────────────────────────────────────────────────────
def test_gc_content_handles_null_sequences():
    dummy = pl.DataFrame({"seq": ["GCGC", None, "ATGC", ""]})
    result = gc_content(dummy)
    # w wyniku powinny pojawić się tylko nie-puste sekwencje
    assert result["count"].sum() == 3
    # Sprawdź, że pusta sekwencja daje 0 %GC
    assert (
        result.filter(pl.col("gc") == 0)["count"].sum() >= 1
    ), "Pusta sekwencja powinna trafiać do koszyka 0 %GC"


# ────────────────────────────────────────────────────────────────────────────────
# 4. Idempotencja przy różnych poziomach równoległości
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("partitions", [1, 2, 8])
def test_gc_content_parallel_consistency(partitions):
    """
    Wynik nie powinien zależeć od ustawienia target_partitions.
    """
    res_a = gc_content(str(FASTQ_PATH), target_partitions=partitions).sort("gc")
    res_b = gc_content(str(FASTQ_PATH), target_partitions=1).sort("gc")
    assert res_a.frame_equal(res_b, null_equal=True)


# ────────────────────────────────────────────────────────────────────────────────
# 5. Szybki test „dymny” pojedynczej sekwencji
# ────────────────────────────────────────────────────────────────────────────────
def test_single_sequence_smoke():
    """
    Upewniamy się, że pojedyncza sztuczna sekwencja zwróci poprawny %GC.
    """
    seq = "GCGTAAcccc"
    df = pl.DataFrame({"seq": [seq]})
    out = gc_content(df)
    expected = _py_gc_percent(seq)
    assert out.height == 1
    assert out.item(0, "gc") == expected
    assert out.item(0, "count") == 1
