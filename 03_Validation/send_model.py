"""Compara modelos pela acurácia de CV, avalia o vencedor e prepara o envio.

Use --dry-run para executar todo o fluxo sem consumir uma submissão.
"""
import argparse
import json
from pathlib import Path

from abalone_csv import DEV_KEY, send_predictions
from sklearn.base import clone

from localrun import select_best_model, save_selection_report
from tratamento import DATA_DIR, load_data, load_training_data, validate_features


def run_abalone_experiment(
    train_path: str | Path = DATA_DIR / "abalone_dataset.csv",
    app_path: str | Path = DATA_DIR / "abalone_app.csv",
    params_path: str | Path | None = None,
    dev_key: str = DEV_KEY, dry_run: bool = False, n_jobs: int = -1,
    n_iter: int = 20, cv_splits: int = 5,
    report_path: str | Path = DATA_DIR / "selection_report.json",
):
    params = None
    if params_path is not None:
        params = json.loads(Path(params_path).read_text(encoding="utf-8"))
        if not isinstance(params, dict) or not params:
            raise ValueError("O arquivo de parâmetros deve conter um objeto JSON não vazio.")
    X, y = load_training_data(train_path)
    data_app = load_data(app_path)
    validate_features(data_app)
    selected, report = select_best_model(
        X, y, cv_splits=cv_splits, n_iter=n_iter, n_jobs=n_jobs, forest_params=params,
    )
    save_selection_report(report, report_path)
    print(f"Relatório salvo em {report_path}")
    model = clone(selected)
    print(f"Reajustando {report['selected_model']} com todos os dados de treino...", flush=True)
    model.fit(X, y)
    predictions = model.predict(data_app)
    print(f"{len(predictions)} previsões geradas.")
    if not dry_run:
        print("Resposta do servidor:\n", send_predictions(predictions, dev_key))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "abalone_dataset.csv")
    parser.add_argument("--app", type=Path, default=DATA_DIR / "abalone_app.csv")
    parser.add_argument("--params", type=Path, help="Parâmetros opcionais para o candidato RandomForest.")
    parser.add_argument("--dev-key", default=DEV_KEY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DATA_DIR / "selection_report.json")
    args = parser.parse_args()
    run_abalone_experiment(
        args.csv, args.app, args.params, args.dev_key, args.dry_run, args.jobs,
        args.n_iter, args.cv, args.report,
    )


if __name__ == "__main__":
    main()
