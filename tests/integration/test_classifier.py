"""tests/unit/test_classifier.py — ML classifier unit tests."""

import os
import tempfile
import pytest
from tests.conftest import synthetic_xy
from core.classifier import RandomForestClassifier
from core.hoeffding import HoeffdingTreeClassifier
from core.constants import ML_FEATURE_DIM


@pytest.fixture
def xy():
    return synthetic_xy(n_each=50)


@pytest.fixture
def clf(xy):
    X, y = xy
    c = RandomForestClassifier(n_trees=5, max_depth=4, random_state=0)
    c.fit(X, y)
    return c


# ─── RandomForestClassifier ───────────────────────────────────────────────────

class TestRandomForest:
    def test_is_trained_after_fit(self, clf):
        assert clf.is_trained

    def test_predict_proba_range(self, clf):
        X, _ = synthetic_xy(n_each=5)
        for x in X:
            p = clf.predict_proba(x)
            assert 0.0 <= p <= 1.0

    def test_predict_binary(self, clf):
        X, _ = synthetic_xy(n_each=5)
        for x in X:
            assert clf.predict(x) in (0, 1)

    def test_untrained_returns_zero(self):
        c = RandomForestClassifier()
        assert c.predict_proba([0.0] * ML_FEATURE_DIM) == 0.0

    def test_human_vectors_score_low(self, clf):
        X, y = synthetic_xy(n_each=20)
        human_X = [x for x, lbl in zip(X, y) if lbl == 0]
        probs = clf.predict_batch(human_X)
        assert sum(p < 0.5 for p in probs) / len(probs) > 0.7

    def test_bot_vectors_score_high(self, clf):
        X, y = synthetic_xy(n_each=20)
        bot_X = [x for x, lbl in zip(X, y) if lbl == 1]
        probs = clf.predict_batch(bot_X)
        assert sum(p >= 0.5 for p in probs) / len(probs) > 0.7

    def test_feature_importances_sum_to_one(self, clf):
        imps = list(clf.all_feature_importances().values())
        assert abs(sum(imps) - 1.0) < 0.01

    def test_top_features_length(self, clf):
        top = clf.get_top_features(5)
        assert len(top) == 5
        assert all(isinstance(n, str) and isinstance(v, float) for n, v in top)

    def test_cross_validate(self, xy):
        X, y = xy
        c = RandomForestClassifier(n_trees=5, max_depth=4, random_state=0)
        c.fit(X, y)
        metrics = c.cross_validate(X, y, k=3)
        assert "f1" in metrics and "far" in metrics and "frr" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_too_few_samples_raises(self):
        c = RandomForestClassifier()
        with pytest.raises(ValueError):
            c.fit([[0.0]*20]*5, [0]*5)

    def test_single_class_raises(self, xy):
        X, _ = xy
        c = RandomForestClassifier(n_trees=3)
        with pytest.raises(ValueError):
            c.fit(X, [0]*len(X))

    def test_save_load(self, clf):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.joblib")
            clf.save_model(path)
            loaded = RandomForestClassifier.load_model(path)
            assert loaded.is_trained
            assert loaded.n_samples == clf.n_samples

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            RandomForestClassifier.load_model("/no/such/file.joblib")


# ─── HoeffdingTreeClassifier ──────────────────────────────────────────────────

class TestHoeffdingTree:
    def test_learn_one_updates_count(self):
        h = HoeffdingTreeClassifier()
        h.learn_one([0.0]*ML_FEATURE_DIM, 0)
        assert h.n_samples == 1
        assert h.is_trained

    def test_predict_proba_range(self):
        h = HoeffdingTreeClassifier()
        X, y = synthetic_xy(n_each=30)
        for x, lbl in zip(X, y):
            h.learn_one(x, lbl)
        for x in X[:10]:
            p = h.predict_proba_one(x)
            assert 0.0 <= p <= 1.0

    def test_untrained_returns_zero(self):
        h = HoeffdingTreeClassifier()
        assert h.predict_proba_one([0.0]*ML_FEATURE_DIM) == 0.0

    def test_batch_fit(self):
        h = HoeffdingTreeClassifier()
        X, y = synthetic_xy(n_each=30)
        h.fit(X, y)
        assert h.n_samples == len(X)

    def test_save_load(self):
        h = HoeffdingTreeClassifier()
        X, y = synthetic_xy(n_each=30)
        for x, lbl in zip(X, y):
            h.learn_one(x, lbl)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.pkl")
            h.save_model(path)
            loaded = HoeffdingTreeClassifier.load_model(path)
            assert loaded.is_trained
            assert loaded.n_samples == h.n_samples

    def test_wrong_dim_raises(self):
        h = HoeffdingTreeClassifier()
        with pytest.raises(ValueError):
            h.learn_one([0.0]*10, 0)
