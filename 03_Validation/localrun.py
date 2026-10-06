"""Seleciona o classificador pela CV de treino e avalia o vencedor no teste reservado.

Busca aleatória limitada; CV aninhada opcional com --nested. Sem SMOTE ou
winsorização por padrão: as classes são equilibradas e nenhum dado é descartado.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from randomforest import build_forest_pipeline, build_forest_search
from tratamento import DATA_DIR, build_preprocessor, load_training_data


def build_knn_pipeline(random_state: int = 42, use_pca: bool = False) -> Pipeline:
    steps = [("prep", build_preprocessor(scale=True))]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=random_state)))
    steps.append(("knn", KNeighborsClassifier()))
    return Pipeline(steps)


def knn_param_grid(max_k: int = 61, step: int = 4) -> dict:
    if max_k < 3 or step < 1:
        raise ValueError("max_k deve ser >= 3 e step deve ser positivo.")
    return {
        "knn__n_neighbors": list(range(3, max_k + 1, step)),
        "knn__p": [1, 2],
        "knn__weights": ["uniform", "distance"],
    }


def build_knn_search(
    X, y, cv_splits: int = 5, n_iter: int = 20,
    random_state: int = 42, n_jobs: int = -1,
) -> RandomizedSearchCV:
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=random_state)
    # Garante k válido até nas menores partições de treino da busca.
    min_train_size = min(len(train) for train, _ in cv.split(X, y))
    if min_train_size < 3:
        raise ValueError("Poucas amostras para a busca KNN (mínimo de 3 por treino interno).")
    grid = knn_param_grid(max_k=min(61, min_train_size))
    candidates = np.prod([len(values) for values in grid.values()])
    return RandomizedSearchCV(
        build_knn_pipeline(random_state), grid, n_iter=min(n_iter, int(candidates)),
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"}, refit="accuracy",
        cv=cv, random_state=random_state, n_jobs=n_jobs, error_score="raise",
    )


def nested_cv_knn(
    X: pd.DataFrame, y: pd.Series, outer_splits: int = 5,
    inner_splits: int = 5, repeats: int = 1, random_state: int = 42,
    n_iter: int = 20, n_jobs: int = -1,
) -> tuple[float, float]:
    """Estima acurácia do procedimento de busca, usando apenas o conjunto de treino."""
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=repeats, random_state=random_state,
    )
    scores = []
    for fold, (train, test) in enumerate(outer_cv.split(X, y), start=1):
        search = build_knn_search(X.iloc[train], y.iloc[train], inner_splits, n_iter, random_state, n_jobs)
        search.fit(X.iloc[train], y.iloc[train])
        score = accuracy_score(y.iloc[test], search.predict(X.iloc[test]))
        scores.append(score)
        print(f"Outer fold {fold}/{outer_cv.get_n_splits()}: acurácia={score:.4f}", flush=True)
    return float(np.mean(scores)), float(np.std(scores))


def compare_baselines(
    X: pd.DataFrame, y: pd.Series, cv_splits: int = 5,
    random_state: int = 42, n_jobs: int = -1,
) -> dict:
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=random_state)
    min_train_size = min(len(train) for train, _ in cv.split(X, y))
    models = {
        "KNN": (KNeighborsClassifier(n_neighbors=min(15, min_train_size), weights="distance"), True),
        "LogReg": (LogisticRegression(max_iter=2000, random_state=random_state), True),
        "HGB": (HistGradientBoostingClassifier(random_state=random_state), False),
        "RandomForest": (RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=random_state), False),
    }
    results = {}
    for name, (model, scale) in models.items():
        pipeline = Pipeline([("prep", build_preprocessor(scale=scale)), ("clf", model)])
        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=["accuracy", "f1_macro"],
            n_jobs=n_jobs, error_score="raise",
        )
        results[name] = scores
        print(
            f"{name}: acurácia={scores['test_accuracy'].mean():.4f} "
            f"± {scores['test_accuracy'].std():.4f}; F1-macro={scores['test_f1_macro'].mean():.4f}",
            flush=True,
        )
    return results


def build_model_searches(X, y, cv_splits=5, n_iter=20, random_state=42, n_jobs=-1, forest_params=None):
    """Mesmas partições e mesma métrica para todos os candidatos."""
    searches = {
        "KNN": build_knn_search(X, y, cv_splits, n_iter, random_state, n_jobs),
        "RandomForest": build_forest_search(n_iter, cv_splits, n_jobs, random_state),
    }
    candidates = {
        "LogReg": (
            Pipeline([("prep", build_preprocessor(scale=True)),
                      ("clf", LogisticRegression(max_iter=2000, random_state=random_state))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0], "clf__class_weight": [None, "balanced"]},
        ),
        "HGB": (
            Pipeline([("prep", build_preprocessor()),
                      ("clf", HistGradientBoostingClassifier(random_state=random_state))]),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31],
             "clf__l2_regularization": [0.0, 1.0, 10.0]},
        ),
    }
    candidates.update({
        "SVM": (
            Pipeline([("prep", build_preprocessor(scale=True)),
                      ("clf", SVC(kernel="rbf", cache_size=256))]),
            {"clf__C": [0.1, 1.0, 10.0, 100.0],
             "clf__gamma": ["scale", 0.001, 0.01, 0.1],
             "clf__class_weight": [None, "balanced"],
             "prep__features__include_ratios": [True, False]},
        ),
        "ExtraTrees": (
            Pipeline([("prep", build_preprocessor()),
                      ("clf", ExtraTreesClassifier(n_estimators=300, n_jobs=1, random_state=random_state))]),
            {"clf__max_features": ["sqrt", 0.7, 1.0],
             "clf__min_samples_leaf": [1, 2, 4, 8],
             "clf__max_depth": [None, 15, 30],
             "prep__features__include_ratios": [True, False]},
        ),
    })
    if forest_params is not None:
        # Um JSON fornecido continua sujeito à comparação; não força o vencedor.
        candidates["RandomForest"] = (build_forest_pipeline(random_state, **forest_params), {})
    for name, (pipeline, params) in candidates.items():
        count = int(np.prod([len(values) for values in params.values()]))
        searches[name] = RandomizedSearchCV(
            pipeline, params, n_iter=min(n_iter, count),
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"}, refit="accuracy",
            cv=StratifiedKFold(cv_splits, shuffle=True, random_state=random_state),
            random_state=random_state, n_jobs=n_jobs, error_score="raise",
        )
    return searches


def select_best_model(X, y, cv_splits=5, n_iter=20, random_state=42, n_jobs=-1, forest_params=None):
    """Retorna pipeline vencedor ajustado no treino e relatório de avaliação.

    O hold-out não participa da seleção. O chamador pode reajustar o pipeline
    em todos os dados somente após esta avaliação para fazer a aplicação.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state,
    )
    searches = build_model_searches(X_train, y_train, cv_splits, n_iter, random_state, n_jobs, forest_params)
    ranking = []
    for name, search in searches.items():
        print(f"Buscando parâmetros de {name}...", flush=True)
        search.fit(X_train, y_train)
        index = search.best_index_
        score = float(search.best_score_)
        if not np.isfinite(score):
            raise ValueError(f"Acurácia de validação inválida para {name}.")
        ranking.append({
            "model": name, "cv_accuracy": score,
            "cv_accuracy_std": float(search.cv_results_["std_test_accuracy"][index]),
            "cv_f1_macro": float(search.cv_results_["mean_test_f1_macro"][index]),
            "params": search.best_params_,
        })
        print(f"{name}: acurácia CV={score:.4f}", flush=True)
    # Em empate exato, mantém a ordem determinística dos candidatos.
    ranking.sort(key=lambda result: result["cv_accuracy"], reverse=True)
    winner = ranking[0]["model"]
    model = searches[winner].best_estimator_
    predictions = model.predict(X_test)
    report = {
        "selected_model": winner, "selection_metric": "cv_accuracy",
        "random_state": random_state, "cv_splits": cv_splits, "n_iter": n_iter,
        "train_rows": len(X_train), "test_rows": len(X_test), "ranking": ranking,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    print(f"Selecionado pela CV: {winner}; acurácia no teste reservado: {report['test_accuracy']:.4f}")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    return model, report


def save_selection_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "abalone_dataset.csv")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--nested", action="store_true", help="Executa também CV aninhada no treino.")
    parser.add_argument("--report", type=Path, default=DATA_DIR / "selection_report.json")
    args = parser.parse_args()
    X, y = load_training_data(args.csv)
    if args.nested:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        mean, std = nested_cv_knn(
            X_train, y_train, outer_splits=args.cv, inner_splits=args.cv,
            n_iter=args.n_iter, n_jobs=args.jobs,
        )
        print(f"CV aninhada KNN: acurácia={mean:.4f} ± {std:.4f}")
    _, report = select_best_model(X, y, args.cv, args.n_iter, n_jobs=args.jobs)
    save_selection_report(report, args.report)
    print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
