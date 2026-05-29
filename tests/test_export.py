"""Tests for CSV/JSON export of BenchmarkRecord lists."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ising_lab.benchmarks import (
    BenchmarkRecord,
    CSV_FIELDS,
    records_from_json,
    records_to_csv,
    records_to_json,
)


def _sample_records():
    return [
        BenchmarkRecord(
            sampler="sa",
            instance_seed=42,
            n=16,
            num_reads=20,
            best_energy=-28.0,
            median_energy=-22.5,
            mean_energy=-21.7,
            wall_time=0.025,
            success_count=4,
            success_prob=0.2,
            tts_99=0.115,
            ground_state_energy=-28.0,
            energies=[-28.0, -22.0, -18.0, -28.0, -22.0],
        ),
        BenchmarkRecord(
            sampler="pt",
            instance_seed=42,
            n=16,
            num_reads=20,
            best_energy=-28.0,
            median_energy=-28.0,
            mean_energy=-27.4,
            wall_time=0.062,
            success_count=18,
            success_prob=0.9,
            tts_99=None,  # represents perfect success (or unrepresentable)
            ground_state_energy=None,  # truth not known
            energies=[-28.0, -28.0, -26.0, -28.0, -28.0],
        ),
    ]


def test_csv_export_columns_and_rowcount(tmp_path: Path):
    records = _sample_records()
    p = records_to_csv(records, tmp_path / "out.csv")
    assert p.exists()

    with p.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == list(CSV_FIELDS)
    assert len(rows) == 2

    # Spot-check scalar values.
    assert rows[0]["sampler"] == "sa"
    assert float(rows[0]["best_energy"]) == pytest.approx(-28.0)
    assert int(rows[0]["success_count"]) == 4
    assert float(rows[0]["success_prob"]) == pytest.approx(0.2)

    # None should serialize to empty cells in CSV.
    assert rows[1]["tts_99"] == ""
    assert rows[1]["ground_state_energy"] == ""


def test_csv_export_omits_energies_histogram(tmp_path: Path):
    """CSV is for the summary; per-read energies belong in JSON."""
    p = records_to_csv(_sample_records(), tmp_path / "x.csv")
    text = p.read_text()
    assert "energies" not in text.splitlines()[0]


def test_json_round_trip_preserves_all_fields(tmp_path: Path):
    """records_from_json(records_to_json(...)) returns equivalent records."""
    src = _sample_records()
    p = records_to_json(src, tmp_path / "out.json")
    restored = records_from_json(p)
    assert len(restored) == len(src)
    for a, b in zip(src, restored):
        assert a == b


def test_json_handles_none_values(tmp_path: Path):
    """None tts_99 and ground_state_energy should survive JSON round-trip."""
    p = records_to_json(_sample_records(), tmp_path / "n.json")
    raw = json.loads(p.read_text())
    assert raw[1]["tts_99"] is None
    assert raw[1]["ground_state_energy"] is None
    restored = records_from_json(p)
    assert restored[1].tts_99 is None
    assert restored[1].ground_state_energy is None
