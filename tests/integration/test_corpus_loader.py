"""
tests/integration/test_corpus_loader.py
=========================================
AttackCorpusLoader integration tests using generated CSV fixtures.
"""

import csv
import os
import pytest
from ingestion.corpus_loader import AttackCorpusLoader, CorpusRecord
from core.constants import ML_FEATURE_DIM


# ─── CSV fixture generators ───────────────────────────────────────────────────

def make_cicids_csv(path: str, n_benign=20, n_attack=20):
    """Write a minimal CICIDS2018-format CSV."""
    cols = [
        "Label","Protocol","Total Fwd Packets","Total Backward Packets",
        "Packet Length Mean","Packet Length Std","Packet Length Variance",
        "Total Length of Fwd Packets","Fwd IAT Mean","Fwd IAT Std",
        "Fwd IAT Min","Fwd IAT Max","Flow Duration","Flow Pkts/s",
        "ACK Flag Count","Init_Win_bytes_forward","Init_Win_bytes_backward",
        "Fwd URG Flags",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for _ in range(n_benign):
            w.writerow({c: ("Benign" if c=="Label" else "100") for c in cols})
        for _ in range(n_attack):
            w.writerow({c: ("DoS Hulk" if c=="Label" else "500") for c in cols})


def make_ctu13_csv(path: str, n_normal=20, n_bot=20):
    """Write a minimal CTU-13 binetflow-format CSV."""
    cols = ["StartTime","Dur","Proto","SrcAddr","Sport","Dir",
            "DstAddr","Dport","State","sTos","dTos",
            "TotPkts","TotBytes","SrcBytes","Label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for _ in range(n_normal):
            w.writerow({c: ("LEGITIMATE" if c=="Label" else "10") for c in cols})
        for _ in range(n_bot):
            w.writerow({c: ("Botnet.Neris" if c=="Label" else "50") for c in cols})


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cicids_file(tmp_path):
    path = str(tmp_path / "test_cicids.csv")
    make_cicids_csv(path, n_benign=30, n_attack=30)
    return path


@pytest.fixture
def ctu13_file(tmp_path):
    path = str(tmp_path / "test_ctu13.csv")
    make_ctu13_csv(path, n_normal=30, n_bot=30)
    return path


@pytest.fixture
def loader():
    return AttackCorpusLoader(max_records_per_file=200, balance_classes=True, verbose=False)


# ─── CICIDS2018 tests ─────────────────────────────────────────────────────────

class TestCICIDS:
    def test_loads_records(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        assert len(records) > 0

    def test_returns_corpus_record_objects(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        assert all(isinstance(r, CorpusRecord) for r in records)

    def test_feature_vector_dimension(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        assert all(len(r.feature_vector) == ML_FEATURE_DIM for r in records)

    def test_labels_are_binary(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        assert all(r.zelvryn_label in (0, 1) for r in records)

    def test_balanced_classes(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        n0 = sum(1 for r in records if r.zelvryn_label == 0)
        n1 = sum(1 for r in records if r.zelvryn_label == 1)
        assert n0 == n1

    def test_source_dataset_label(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        assert all(r.source_dataset == "cicids2018" for r in records)

    def test_missing_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_cicids2018("/no/such/file.csv")

    def test_feature_vectors_finite(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        for r in records:
            assert all(not (v != v) and abs(v) != float("inf") for v in r.feature_vector)


# ─── CTU-13 tests ─────────────────────────────────────────────────────────────

class TestCTU13:
    def test_loads_records(self, loader, ctu13_file):
        records = loader.load_ctu13(ctu13_file)
        assert len(records) > 0

    def test_feature_vector_dimension(self, loader, ctu13_file):
        records = loader.load_ctu13(ctu13_file)
        assert all(len(r.feature_vector) == ML_FEATURE_DIM for r in records)

    def test_botnet_labeled_as_one(self, loader, ctu13_file):
        records = loader.load_ctu13(ctu13_file)
        for r in records:
            if "Botnet" in r.original_label:
                assert r.zelvryn_label == 1

    def test_normal_labeled_as_zero(self, loader, ctu13_file):
        records = loader.load_ctu13(ctu13_file)
        for r in records:
            if r.original_label in ("LEGITIMATE","Normal","Background"):
                assert r.zelvryn_label == 0


# ─── Training arrays and persistence ──────────────────────────────────────────

class TestTrainingArrays:
    def test_to_training_arrays(self, loader, cicids_file):
        records = loader.load_cicids2018(cicids_file)
        X, y = loader.to_training_arrays(records)
        assert len(X) == len(y) == len(records)

    def test_save_and_load_processed(self, loader, cicids_file, tmp_path):
        records = loader.load_cicids2018(cicids_file)
        loader.save_processed(records, output_dir=str(tmp_path))
        X2, y2 = AttackCorpusLoader.load_processed(str(tmp_path))
        assert len(X2) == len(records)

    def test_load_processed_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AttackCorpusLoader.load_processed(str(tmp_path / "empty"))
