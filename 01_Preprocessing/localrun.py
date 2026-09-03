
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
 
# ----------------------------------------------------------------------
# CONFIGURAÇÕES — ajuste o nome do arquivo se for testar outra versão
# ----------------------------------------------------------------------
ARQUIVO = 'diabetes_dataset_preprocess.xlsx'
FEATURE_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
N_REPETICOES = 10  # roda várias vezes com seeds diferentes p/ não confiar num único split
 
print(f'\n - Lendo o arquivo pré-processado: {ARQUIVO}')
data = pd.read_excel(ARQUIVO)
 
print("\nColunas encontradas no arquivo:")
print(data.columns.tolist())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
 
# ----------------------------------------------------------------------
# CONFIGURAÇÕES — ajuste o nome do arquivo se for testar outra versão
# ----------------------------------------------------------------------
ARQUIVO = 'diabetes_dataset_preprocess.xlsx'
FEATURE_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
N_REPETICOES = 10  # roda várias vezes com seeds diferentes p/ não confiar num único split
 
print(f'\n - Lendo o arquivo pré-processado: {ARQUIVO}')
data = pd.read_excel(ARQUIVO)
 
print("\nColunas encontradas no arquivo:")
print(data.columns.tolist())
 
# Verificação: garante que feature_cols bate com o que existe no arquivo
faltando = [c for c in FEATURE_COLS if c not in data.columns]
if faltando:
    raise ValueError(f'Essas colunas não existem no arquivo: {faltando}')
 
print(' - Separando features (X) e target (y)')
X = data[FEATURE_COLS]
y = data.Outcome
 
# Rodando múltiplos splits 80/20 pra ter uma média + desvio, em vez de
# confiar num único split (que pode dar sorte ou azar). Sem seed fixa:
# cada execução do script gera splits diferentes.
acuracias = []
for _ in range(N_REPETICOES):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acuracias.append(acc)
 
acuracias_arr = pd.Series(acuracias)
 
print("\n----------- RESULTADO (um split aleatório de exemplo) -----------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
acc_unico = accuracy_score(y_test, y_pred)
print(f'Dimensões -> Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras')
print(f"A acurácia do modelo (um único split) é: {acc_unico * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Não diabético', 'Diabético']))
 
print("\n----------- RESULTADO (média de {} splits diferentes) -----------".format(N_REPETICOES))
print(f"Acurácia média: {acuracias_arr.mean()*100:.2f}%")
print(f"Desvio padrão:  {acuracias_arr.std()*100:.2f}%")
print(f"Mínimo / Máximo: {acuracias_arr.min()*100:.2f}% / {acuracias_arr.max()*100:.2f}%")
print("---------------------------------")
# Verificação: garante que feature_cols bate com o que existe no arquivo
faltando = [c for c in FEATURE_COLS if c not in data.columns]
if faltando:
    raise ValueError(f'Essas colunas não existem no arquivo: {faltando}')
 
print(' - Separando features (X) e target (y)')
X = data[FEATURE_COLS]
y = data.Outcome
 
# ----------------------------------------------------------------------
# Rodando múltiplos splits 80/20 pra ter uma média + desvio, em vez de
# confiar num único random_state (que pode dar sorte ou azar)
# ----------------------------------------------------------------------
acuracias = []
for seed in range(N_REPETICOES):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acuracias.append(acc)
 
acuracias_arr = pd.Series(acuracias)
 
print("\n----------- RESULTADO (holdout único, seed=42, igual ao script original) -----------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
acc_unico = accuracy_score(y_test, y_pred)
print(f'Dimensões -> Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras')
print(f"A acurácia do modelo (um único split) é: {acc_unico * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Não diabético', 'Diabético']))
 
print("\n----------- RESULTADO (média de {} splits diferentes) -----------".format(N_REPETICOES))
print(f"Acurácia média: {acuracias_arr.mean()*100:.2f}%")
print(f"Desvio padrão:  {acuracias_arr.std()*100:.2f}%")
print(f"Mínimo / Máximo: {acuracias_arr.min()*100:.2f}% / {acuracias_arr.max()*100:.2f}%")
print("\n(A média de várias repetições é uma estimativa mais confiável do que um único split.)")
print("(Lembrete: este número pode estar levemente otimista por causa da normalização")
print(" calculada em todo o arquivo antes do split — ver comentário no topo do script.)")
print("---------------------------------")