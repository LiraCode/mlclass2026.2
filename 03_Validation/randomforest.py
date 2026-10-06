"""Busca Random Forest com teste reservado e pré-processamento dentro da CV."""
import argparse
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from tratamento import DATA_DIR, build_preprocessor, load_training_data


def build_forest_pipeline(random_state: int = 42, n_jobs: int = 1, **params) -> Pipeline:
    forest = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    forest.set_params(**params)
    return Pipeline([("prep", build_preprocessor()), ("rf", forest)])


def build_forest_search(n_iter=20, cv_splits=5, n_jobs=-1, random_state=42):
    # Paraleliza a busca; cada floresta usa uma thread para evitar paralelismo aninhado.
    return RandomizedSearchCV(
        build_forest_pipeline(random_state),
        {
            "rf__n_estimators": [200, 400, 600],
            "rf__max_depth": [None, 10, 20, 30],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4, 8],
            "rf__max_features": ["sqrt", 0.7, 1.0],
            "rf__class_weight": [None, "balanced_subsample"],
        },
        n_iter=n_iter,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        refit="accuracy",  # Métrica usada pelo servidor da atividade.
        cv=StratifiedKFold(cv_splits, shuffle=True, random_state=random_state),
        random_state=random_state, n_jobs=n_jobs, error_score="raise", verbose=1,
    )


def run_abalone_experiment(
    csv_path: str | Path = DATA_DIR / "abalone_dataset.csv",
    n_iter: int = 20, cv_splits: int = 5, n_jobs: int = -1,
    random_state: int = 42, params_path: str | Path | None = None,
):
    X, y = load_training_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state,
    )
    search = build_forest_search(n_iter, cv_splits, n_jobs, random_state)
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    print("Melhores parâmetros:", search.best_params_)
    print(f"Acurácia média na CV de treino: {search.best_score_:.4f}")
    print(f"Acurácia no teste reservado: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    print("Matriz de confusão (classes 1, 2, 3):\n", confusion_matrix(y_test, predictions, labels=[1, 2, 3]))
    if params_path is not None:
        path = Path(params_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {key.removeprefix("rf__"): value for key, value in search.best_params_.items()}
        path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        print(f"Parâmetros para forestsent.py salvos em {path}")
    return search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "abalone_dataset.csv")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--params-output", type=Path, default=DATA_DIR / "forest_params.json")
    args = parser.parse_args()
    run_abalone_experiment(args.csv, args.n_iter, args.cv, args.jobs, params_path=args.params_output)


if __name__ == "__main__":
    main()
