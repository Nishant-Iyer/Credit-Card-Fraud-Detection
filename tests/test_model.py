"""Basic tests for the fraud detection pipeline."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import load_data
from model_utils import train_and_evaluate


def test_training_pipeline(tmp_path: Path) -> None:
    X, y = load_data(None)
    model, report, auc = train_and_evaluate(X, y)
    assert "precision" in report
    assert 0.0 <= auc <= 1.0
    out_path = tmp_path / "model.joblib"
    import joblib

    joblib.dump(model, out_path)
    assert out_path.exists()
