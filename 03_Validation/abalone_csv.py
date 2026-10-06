#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Treina KNN para Abalone e envia previsões para verificação no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""
import argparse
from pathlib import Path

import pandas as pd
import requests
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from tratamento import DATA_DIR, build_preprocessor, load_data, load_training_data, validate_features

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "Delta"


def send_predictions(predictions, dev_key: str = DEV_KEY) -> str:
    predictions = pd.Series(predictions)
    if predictions.empty or not predictions.isin([1, 2, 3]).all():
        raise ValueError("Previsões devem conter classes 1, 2 ou 3, sem ausências.")
    data = {"dev_key": dev_key, "predictions": predictions.astype(int).to_json(orient="values")}
    response = requests.post(url=URL, data=data, timeout=30)
    response.raise_for_status()
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "abalone_dataset.csv")
    parser.add_argument("--app", type=Path, default=DATA_DIR / "abalone_app.csv")
    parser.add_argument("--dev-key", default=DEV_KEY)
    parser.add_argument("--dry-run", action="store_true", help="Valida as previsões sem enviar ao servidor.")
    args = parser.parse_args()
    print(" - Lendo o dataset Abalone e criando o modelo preditivo")
    X, y = load_training_data(args.csv)
    # O mesmo pipeline trata treino e aplicação, incluindo escala e categorias.
    neigh = Pipeline([("prep", build_preprocessor(scale=True)), ("knn", KNeighborsClassifier(n_neighbors=3))])
    neigh.fit(X, y)
    data_app = load_data(args.app)
    validate_features(data_app)
    predictions = neigh.predict(data_app)
    print(f" - {len(predictions)} previsões geradas")
    if not args.dry_run:
        print(" - Resposta do servidor:\n", send_predictions(predictions, args.dev_key))


if __name__ == "__main__":
    main()
