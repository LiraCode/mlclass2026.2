"""Leitura e pré-processamento compartilhados pelos experimentos Abalone.

A engenharia de atributos preserva linhas. Imputação e escala são aprendidas
pelo pipeline exclusivamente no treino de cada partição da validação.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

DATA_DIR = Path(__file__).resolve().parent
NUMERIC_COLUMNS = [
    "length", "diameter", "height", "whole_weight", "shucked_weight",
    "viscera_weight", "shell_weight",
]
FEATURE_COLUMNS = ["bmi", "length_dia_ratio", "meat_yield", "shell_ratio"]
RAW_COLUMNS = ["sex", *NUMERIC_COLUMNS]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Lê CSV; erros de leitura são propagados com seu contexto original."""
    return pd.read_csv(file_path)


def save_data(df: pd.DataFrame, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_features(df: pd.DataFrame) -> None:
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    if df.empty:
        raise ValueError("O CSV não contém amostras.")


def load_training_data(file_path: str | Path = DATA_DIR / "abalone_dataset.csv"):
    df = load_data(file_path)
    validate_features(df)
    if "type" not in df:
        raise ValueError("CSV de treino deve conter o alvo 'type'; abalone_app.csv é só para aplicação.")
    if df["type"].isna().any() or not df["type"].isin([1, 2, 3]).all():
        raise ValueError("O alvo 'type' deve conter apenas as classes 1, 2 e 3, sem ausências.")
    return df[RAW_COLUMNS].copy(), df["type"].astype(int)


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer.div(denom.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def add_features(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """Mantém índice e número de linhas, deixando divisões inválidas para imputação."""
    validate_features(df)
    result = df.copy()
    result[NUMERIC_COLUMNS] = result[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    result[NUMERIC_COLUMNS] = result[NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan)
    result["bmi"] = _safe_div(result["whole_weight"], result["height"] ** 2)
    result["length_dia_ratio"] = _safe_div(result["length"], result["diameter"])
    result["meat_yield"] = _safe_div(result["shucked_weight"], result["whole_weight"])
    result["shell_ratio"] = _safe_div(result["shell_weight"], result["whole_weight"])
    if drop_original:
        result = result.drop(columns=[c for c in NUMERIC_COLUMNS if c != "viscera_weight"])
    return result


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Transformação determinística, sem incluir alvo ou colunas extras no modelo."""

    def __init__(self, drop_original: bool = False, include_ratios: bool = True):
        self.drop_original = drop_original
        self.include_ratios = include_ratios

    def fit(self, X: pd.DataFrame, y=None):
        validate_features(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        validate_features(X)
        result = add_features(X[RAW_COLUMNS], drop_original=self.drop_original)
        return result if self.include_ratios else result.drop(columns=FEATURE_COLUMNS)


def build_preprocessor(scale: bool = False, drop_original: bool = False) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    columns = ColumnTransformer([
        ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["sex"]),
    ])
    return Pipeline([("features", FeatureEngineer(drop_original)), ("columns", columns)])


def main() -> None:
    """Exporta apenas atributos determinísticos; não ajusta estatísticas no CSV inteiro."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "abalone_dataset.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "preprocessed_dataset.csv")
    args = parser.parse_args()
    data = add_features(load_data(args.csv))
    save_data(data, args.output)
    print(f"{len(data)} linhas salvas em {args.output}. Imputação e escala ficam no pipeline de treino.")


if __name__ == "__main__":
    main()
