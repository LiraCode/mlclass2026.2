"""Regressões para alinhamento, isolamento do treino e contrato de envio."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abalone_csv import URL, send_predictions
from send_model import run_abalone_experiment
from localrun import build_knn_search, nested_cv_knn, select_best_model
from randomforest import build_forest_pipeline
from tratamento import DATA_DIR, FeatureEngineer, add_features, build_preprocessor, load_training_data


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_training_data()

    def test_zero_denominators_preserve_rows_index_and_target(self):
        data = self.X.iloc[:12].copy()
        data.loc[data.index[:3], ["height", "diameter", "whole_weight"]] = 0
        original = data.copy(deep=True)
        features = add_features(data)
        pd.testing.assert_index_equal(features.index, data.index)
        pd.testing.assert_frame_equal(data, original)
        self.assertTrue(features.iloc[:3][["bmi", "length_dia_ratio", "meat_yield", "shell_ratio"]].isna().all().all())
        model = build_forest_pipeline(n_estimators=5).fit(data, self.y.iloc[:12])
        self.assertEqual(len(model.predict(data)), len(data))

    def test_feature_engineer_excludes_target_and_honors_drop_original(self):
        data = self.X.iloc[:10].assign(type=self.y.iloc[:10], extra=123)
        result = clone(FeatureEngineer(drop_original=True)).fit_transform(data)
        self.assertNotIn("type", result)
        self.assertNotIn("extra", result)
        self.assertNotIn("length", result)
        self.assertIn("viscera_weight", result)

    def test_imputation_is_learned_only_from_train(self):
        train = self.X.iloc[:20].copy()
        train.loc[train.index[0], "length"] = np.nan
        prep = build_preprocessor().fit(train)
        imputer = prep.named_steps["columns"].named_transformers_["num"].named_steps["imputer"]
        learned = imputer.statistics_.copy()
        self.assertAlmostEqual(learned[0], train["length"].median())
        app = self.X.iloc[20:25].copy()
        app["length"] = 1e6
        prep.transform(app)
        np.testing.assert_array_equal(learned, imputer.statistics_)

    def test_unknown_categories_and_empty_numeric_feature(self):
        train = self.X.iloc[:20].copy()
        train["height"] = 0
        prep = build_preprocessor(scale=True).fit(train)
        app = train.iloc[:3].copy()
        app["sex"] = ["unknown", np.nan, "M"]
        transformed = prep.transform(app)
        self.assertEqual(transformed.shape[0], len(app))
        self.assertTrue(np.isfinite(transformed).all())

    def test_optional_ratios_keep_raw_features_and_rows(self):
        train = self.X.iloc[:30].copy()
        train.loc[train.index[0], "height"] = 0
        with_ratios = build_preprocessor(scale=True).fit_transform(train)
        prep = build_preprocessor(scale=True).set_params(features__include_ratios=False)
        without_ratios = clone(prep).fit_transform(train)
        self.assertEqual(with_ratios.shape[1] - without_ratios.shape[1], 4)
        self.assertEqual(without_ratios.shape[0], len(train))
        self.assertTrue(np.isfinite(without_ratios).all())
        transformed = prep.named_steps["features"].fit_transform(train)
        self.assertIn("height", transformed)
        self.assertNotIn("bmi", transformed)

    def test_application_csv_cannot_be_used_for_training(self):
        with self.assertRaisesRegex(ValueError, "alvo 'type'"):
            load_training_data(DATA_DIR / "abalone_app.csv")

    def test_invalid_target_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            self.X.iloc[:3].assign(type=[1, np.nan, 3]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "classes 1, 2 e 3"):
                load_training_data(path)

    def test_knn_search_and_nested_cv_on_small_partitions(self):
        indices = self.y.groupby(self.y).head(8).index
        X, y = self.X.loc[indices], self.y.loc[indices]
        search = build_knn_search(X, y, cv_splits=2, n_iter=1, n_jobs=1).fit(X, y)
        self.assertEqual(len(search.predict(X)), len(y))
        mean, std = nested_cv_knn(X, y, outer_splits=2, inner_splits=2, n_iter=1, n_jobs=1)
        self.assertTrue(0 <= mean <= 1)
        self.assertGreaterEqual(std, 0)

    @patch("localrun.build_model_searches")
    def test_selection_uses_cv_and_keeps_holdout_out_of_search(self, build_searches):
        searches = {}
        for name, score in [("RandomForest", 0.7), ("LogReg", 0.9)]:
            search = Mock(best_score_=score, best_index_=0, best_params_={})
            search.cv_results_ = {"std_test_accuracy": [0.01], "mean_test_f1_macro": [score]}
            search.best_estimator_.predict.side_effect = lambda X: np.ones(len(X), dtype=int)
            searches[name] = search
        build_searches.return_value = searches
        model, report = select_best_model(self.X, self.y, n_iter=1, cv_splits=2, n_jobs=1)
        self.assertIs(model, searches["LogReg"].best_estimator_)
        self.assertEqual(report["selected_model"], "LogReg")
        train = searches["LogReg"].fit.call_args.args[0]
        test = model.predict.call_args.args[0]
        self.assertFalse(set(train.index) & set(test.index))
        self.assertEqual(len(train) + len(test), len(self.X))
        for search in searches.values():
            pd.testing.assert_index_equal(search.fit.call_args.args[0].index, train.index)
        searches["RandomForest"].best_estimator_.predict.assert_not_called()

    @patch("send_model.send_predictions", return_value="ok")
    @patch("send_model.clone")
    @patch("send_model.select_best_model")
    def test_submission_uses_refitted_winner(self, select, clone_model, send):
        selected = Mock()
        select.return_value = (selected, {"selected_model": "LogReg"})
        model = clone_model.return_value
        expected = np.ones(1045, dtype=int)
        model.predict.return_value = expected
        with tempfile.TemporaryDirectory() as directory:
            result = run_abalone_experiment(
                dry_run=False, report_path=Path(directory) / "report.json",
            )
        clone_model.assert_called_once_with(selected)
        self.assertEqual(len(model.fit.call_args.args[0]), len(self.X))
        self.assertIs(result, expected)
        send.assert_called_once()
        self.assertIs(send.call_args.args[0], expected)

    @patch("send_model.select_best_model", side_effect=ValueError("CV falhou"))
    @patch("abalone_csv.requests.post")
    def test_validation_failure_prevents_submission(self, post, select):
        with self.assertRaisesRegex(ValueError, "CV falhou"):
            run_abalone_experiment(dry_run=False)
        post.assert_not_called()

    @patch("abalone_csv.requests.post")
    def test_send_contract(self, post):
        post.return_value = Mock(text="ok")
        self.assertEqual(send_predictions(np.array([1, 3, 2]), "equipe"), "ok")
        post.assert_called_once_with(
            url=URL, data={"dev_key": "equipe", "predictions": "[1,3,2]"}, timeout=30,
        )
        post.return_value.raise_for_status.assert_called_once()

    @patch("abalone_csv.requests.post")
    def test_invalid_predictions_never_sent(self, post):
        for predictions in ([], [1, np.nan], [0, 4]):
            with self.assertRaises(ValueError):
                send_predictions(predictions)
        post.assert_not_called()

    @patch("abalone_csv.requests.post")
    def test_imports_have_no_network_side_effects(self, post):
        for module in ("abalone_csv", "send_model", "randomforest", "localrun", "tratamento"):
            importlib.reload(sys.modules[module])
        post.assert_not_called()

    @patch("abalone_csv.requests.post")
    def test_selection_dry_run_predicts_every_application_row(self, post):
        with tempfile.TemporaryDirectory() as directory:
            params_path = Path(directory) / "params.json"
            params_path.write_text(json.dumps({"n_estimators": 5}), encoding="utf-8")
            predictions = run_abalone_experiment(
                params_path=params_path, dry_run=True, n_jobs=1, n_iter=1, cv_splits=2,
                report_path=Path(directory) / "report.json",
            )
        self.assertEqual(len(predictions), len(pd.read_csv(DATA_DIR / "abalone_app.csv")))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
