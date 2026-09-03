
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
 
# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
ARQUIVO_ENTRADA = 'diabetes_dataset.xlsx'
ARQUIVO_SAIDA = 'diabetes_dataset_preprocess.xlsx'
 
ARQUIVO_APP = 'diabetes_app.xlsx'
ARQUIVO_APP_SAIDA = 'diabetes_app_preprocess.xlsx' 
NORMALIZAR = True  # para testar a diferença entre normalizar ou não.
 
LIMIAR_EXCLUSAO = 0.40  # mínimo para questionar exclusão de coluna (% de dados ausentes).
 
# Colunas em que 0 é FISIOLOGICAMENTE IMPOSSÍVEL.
COLUNAS_ZERO_INVALIDO = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
 
# Coluna alvo (variável dependente) do dataset.
COLUNA_ALVO = 'Outcome'

#colunas em que 0 é possivel (não será tratado como ausente).
COLUNAS_ZERO_VALIDO = ['Pregnancies', COLUNA_ALVO]
 

 # ----------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------
 
def calcular_percentual_ausente(df):
    """Retorna uma Series com o percentual (0-100) de NaN por coluna."""
    return df.isna().mean() * 100
 
 
def perguntar_exclusao_coluna(col, pct_ausente):
    """Pergunta ao usuário (via terminal) se deseja excluir a coluna."""
    print(f'\n   [ATENÇÃO] A coluna "{col}" possui {pct_ausente:.2f}% de dados '
          f'ausentes (acima do limite de {LIMIAR_EXCLUSAO * 100:.0f}%).')
    while True:
        resposta = input(f'   Deseja EXCLUIR a coluna "{col}" do dataset final? [s/n]: ').strip().lower()
        if resposta in ('s', 'sim', 'y', 'yes'):
            return True
        if resposta in ('n', 'nao', 'não', 'no'):
            return False
        print('   Resposta inválida, digite "s" ou "n".')
 
 
def tratar_zeros_invalidos(df):
    """Converte zeros fisiologicamente impossíveis em NaN."""
    colunas_alvo = [c for c in COLUNAS_ZERO_INVALIDO if c in df.columns]
    for col in colunas_alvo:
        n_zeros = int((df[col] == 0).sum())
        if n_zeros > 0:
            print(f'   -> Coluna "{col}": {n_zeros} valor(es) igual a 0 '
                  f'tratado(s) como ausente (fisiologicamente impossível).')
            df.loc[df[col] == 0, col] = np.nan
    return df
 
 
def preencher_valores_ausentes(df, rng):
    """
    Preenche cada NaN sorteando um valor real já observado (válido) na
    própria coluna. Preserva média, desvio-padrão e formato da
    distribuição original, evitando o viés de usar média/mediana fixa
    ou um valor de min/max fixo.
    """
    for col in df.columns:
        n_ausentes = int(df[col].isna().sum())
        if n_ausentes == 0:
            continue
 
        valores_validos = df.loc[df[col].notna(), col].values
        if len(valores_validos) == 0:
            print(f'   [AVISO] Coluna "{col}" não possui nenhum valor válido. '
                  f' pulando.')
            continue
 
        valores_sorteados = rng.choice(valores_validos, size=n_ausentes, replace=True)
        df.loc[df[col].isna(), col] = valores_sorteados
 
        print(f'   -> Coluna "{col}": {n_ausentes} valor(es) ausente(s) preenchido(s) '
              f'por amostragem da distribuição real '
              f'(min={valores_validos.min():.2f}, max={valores_validos.max():.2f}).')
 
    return df
 
 
def normalizar_features(df, colunas_features, params=None):

    df = df.copy()
    calculando = params is None
    if calculando:
        params = RobustScaler()
        df[colunas_features] = params.fit_transform(df[colunas_features])
        print('   -> Normalizando dataset.')
    else:
        df[colunas_features] = params.transform(df[colunas_features])
        print('   -> aplicando a normalização no diabetes_app.')
 
    return df, params
 
 
def processar_app(colunas_finais, params_normalizacao, rng):
    """
    Aplica ao diabetes_app.xlsx as MESMAS transformações do treino:
    mesmas colunas, mesmo tratamento de zero inválido, mesma imputação
    (por amostragem) e a MESMA normalização (reaproveitando mediana/IQR
    calculados no treino, nunca recalculando no app).
    """
    print(f'\n - Lendo o arquivo de aplicação: {ARQUIVO_APP}')
    app = pd.read_excel(ARQUIVO_APP)
    print(f'   Dimensões: {app.shape[0]} linhas x {app.shape[1]} colunas')
 
    faltando = [c for c in colunas_finais if c not in app.columns]
    if faltando:
        print(f'   [AVISO] Colunas esperadas ausentes no app: {faltando}. Pulando processamento do app.')
        return
 
    app = app[colunas_finais].copy()
 
 
    if NORMALIZAR:
        print('   - Normalizando o app com os parâmetros do TREINO (não recalculados)')
        app, _ = normalizar_features(app, colunas_finais, params=params_normalizacao)
 
    app.to_excel(ARQUIVO_APP_SAIDA, index=False)
    print(f'   - diabetes_app pré-processado salvo em: {ARQUIVO_APP_SAIDA}')

 
#--------------------------------------------------------------
# Função principal
#--------------------------------------------------------------
 
def main():
    rng = np.random.default_rng()
 
    print(f'\n - Lendo o dataset original: {ARQUIVO_ENTRADA}')
    df = pd.read_excel(ARQUIVO_ENTRADA)
    print(f'   Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas')
    print(f'   Colunas encontradas: {list(df.columns)}')
 
    # --------------------------------------------------------------
    # ETAPA 1 - Percentual de dados ausentes por coluna
    # --------------------------------------------------------------
    print('\n - Analisando percentual de dados ausentes por coluna')
    pct_ausente = calcular_percentual_ausente(df)
    for col, pct in pct_ausente.items():
        print(f'   {col}: {pct:.2f}%')
 
    colunas_para_excluir = []
    for col, pct in pct_ausente.items():
        if col == COLUNA_ALVO:
            continue  # nunca oferece excluir a variável alvo
        if pct > LIMIAR_EXCLUSAO * 100:
            if perguntar_exclusao_coluna(col, pct):
                colunas_para_excluir.append(col)
 
    if colunas_para_excluir:
        print(f'\n   Colunas removidas do dataset final: {colunas_para_excluir}')
        df = df.drop(columns=colunas_para_excluir)
    else:
        print('\n   Nenhuma coluna ultrapassou o limite (ou o usuário optou por manter todas).')
 
    # --------------------------------------------------------------
    # ETAPA 2 - Zeros fisiologicamente impossíveis -> NaN
    # --------------------------------------------------------------
    print('\n - Verificando valores zerados fisiologicamente inválidos')
    df = tratar_zeros_invalidos(df)
 
    # --------------------------------------------------------------
    # ETAPA 3 - Preenchimento (imputação) dos valores ausentes
    # --------------------------------------------------------------
    print('\n - Preenchendo valores ausentes')
    df = preencher_valores_ausentes(df, rng)
 
    # --------------------------------------------------------------
    # ETAPA 4 - Calcular parâmetros de normalização no TREINO e aplicar
    # --------------------------------------------------------------
    colunas_features_finais = [c for c in df.columns if c != COLUNA_ALVO]
    params_normalizacao = None
    if NORMALIZAR:
        print('\n - Calculando parâmetros de normalização (mediana/IQR) no treino e aplicando')
        df, params_normalizacao = normalizar_features(df, colunas_features_finais)
    else:
        print('\n - Normalização desativada (NORMALIZAR=False)')
 
    # --------------------------------------------------------------
    # RELATÓRIO FINAL (treino)
    # --------------------------------------------------------------
    print('\n - Verificação final de dados ausentes por coluna (treino):')
    print(df.isna().sum().to_string())
 
    # --------------------------------------------------------------
    # SALVANDO O DATASET DE TREINO PRÉ-PROCESSADO
    # --------------------------------------------------------------
    df.to_excel(ARQUIVO_SAIDA, index=False)
    print(f'\n - Dataset de treino pré-processado salvo em: {ARQUIVO_SAIDA}')
 
    # --------------------------------------------------------------
    # ETAPA 5 - Aplicar as MESMAS transformações (mesmas colunas, mesma
    # normalização calculada no treino) ao diabetes_app.xlsx
    # --------------------------------------------------------------
    processar_app(colunas_features_finais, params_normalizacao, rng)
 
    print('\n - Pronto: rode o diabetes_xlsx.py.')
 
 
if __name__ == '__main__':
    main()