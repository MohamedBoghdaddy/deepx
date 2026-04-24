import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import (  # noqa: E402
    TRAINING_MANIFEST_FILENAME,
    TRAINING_STATE_FILENAME,
    ensure_trained_model,
    is_training_state_compatible,
    load_training_state,
    save_training_manifest,
    save_training_state,
)


class TrainingCheckpointTests(unittest.TestCase):
    def test_training_state_round_trip_and_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.csv"
            validation_path = root / "validation.csv"
            checkpoint_path = root / "model.pt"
            state_path = root / TRAINING_STATE_FILENAME

            train_path.write_text("review_id,review_text\n1,test\n", encoding="utf-8")
            validation_path.write_text("review_id,review_text\n2,test\n", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            batch = torch.randn(2, 4)
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            save_training_state(
                state_path=state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_name="arabert",
                config={"num_epochs": 3, "batch_size": 8},
                checkpoint_path=checkpoint_path,
                train_path=train_path,
                validation_path=validation_path,
                next_epoch=2,
                best_f1=0.71,
                best_threshold=0.45,
                best_metrics={"micro_f1": 0.71},
                best_threshold_history={"0.45": {"micro_f1": 0.71}},
                stagnant_epochs=1,
                class_distribution={"num_samples": 1},
            )

            training_state = load_training_state(state_path)
            self.assertEqual(training_state["next_epoch"], 2)
            self.assertAlmostEqual(training_state["best_threshold"], 0.45)

            compatible, reasons = is_training_state_compatible(
                training_state=training_state,
                checkpoint_path=checkpoint_path,
                model_name="arabert",
                config={"num_epochs": 3, "batch_size": 8},
                train_path=train_path,
                validation_path=validation_path,
            )
            self.assertTrue(compatible)
            self.assertEqual(reasons, [])

            incompatible, blockers = is_training_state_compatible(
                training_state=training_state,
                checkpoint_path=checkpoint_path,
                model_name="arabert",
                config={"num_epochs": 5, "batch_size": 8},
                train_path=train_path,
                validation_path=validation_path,
            )
            self.assertFalse(incompatible)
            self.assertIn("Training configuration changed.", blockers)

    def test_ensure_trained_model_can_fallback_to_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / "model.pt"

            torch.save(
                {
                    "model_name": "existing-checkpoint-model",
                    "config": {"num_epochs": 1, "batch_size": 4},
                    "metrics": {"micro_f1": 0.66},
                    "best_threshold": 0.4,
                },
                checkpoint_path,
            )

            result = ensure_trained_model(
                train_path=root / "missing_train.csv",
                validation_path=root / "missing_validation.csv",
                model_name="arabert",
                config={"num_epochs": 5},
                output_dir=output_dir,
                allow_checkpoint_fallback=True,
            )

            self.assertTrue(result["reused_existing_training"])
            self.assertFalse(result["compatibility_verified"])
            self.assertEqual(Path(result["checkpoint_path"]), checkpoint_path.resolve())
            self.assertAlmostEqual(result["best_threshold"], 0.4)
            self.assertIn("Training manifest not found", " ".join(result["retrain_reasons"]))

    def test_incompatible_manifest_does_not_trigger_checkpoint_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / "model.pt"
            manifest_path = output_dir / TRAINING_MANIFEST_FILENAME
            train_path = root / "train.csv"
            validation_path = root / "validation.csv"

            train_path.write_text("review_id,review_text\n1,test\n", encoding="utf-8")
            validation_path.write_text("review_id,review_text\n2,test\n", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            save_training_manifest(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                model_name="arabert",
                config={"num_epochs": 1},
                metrics={"micro_f1": 0.5},
                threshold=0.5,
                train_path=train_path,
                validation_path=validation_path,
            )
            self.assertTrue(manifest_path.exists())

            with mock.patch("train.load_dataframe", return_value=pd.DataFrame()) as load_dataframe_mock:
                with mock.patch("train.train_model", return_value=(object(), {"micro_f1": 0.1}, 0.5)) as train_model_mock:
                    result = ensure_trained_model(
                        train_path=train_path,
                        validation_path=validation_path,
                        model_name="marbert",
                        config={"num_epochs": 1},
                        output_dir=output_dir,
                        allow_checkpoint_fallback=True,
                    )

            self.assertFalse(result["reused_existing_training"])
            self.assertTrue(result["compatibility_verified"])
            train_model_mock.assert_called_once()
            self.assertEqual(load_dataframe_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
