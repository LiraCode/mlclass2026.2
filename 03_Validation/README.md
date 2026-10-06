# Atividade 03 - Avaliação de classificadores

Para tudo que nos é enviado, assumimos que você está seguindo o código de honra a seguir.

## Código de Honra

>"Como membro da comunidade deste curso, não vou participar nem tolerar a desonestidade acadêmica".

## Objetivo da atividade
*Trabalhar a metodologia e as técnicas para a avaliação de classificadores*

## Descrição da atividade
A atividade da equipe consiste em construir e validar um modelo(s) preditivo(s) com o intuito de garantir o seu poder de generalização, utilizando a metodologia e técnicas vistas em sala de aula.

Nessa atividade vocês poderão utilizar qualquer algoritmo de aprendizagem mesmo os ainda não vistos em sala de aula tais como: SVM, Redes Neurais, RBFs, etc. Um detalhe importante é que o entendimento do algoritmo é necessário pois as equipes melhores ranqueadas terão, como nas outras atividades, compartilhar o que foi aprendido e isso inclui o algoritmo utilizado.

Para o envio da atividade poderão ser utilizados os mesmos modelos de programas para ler e enviar os resultados utilizados na Atividade 01 - Pré-processamento, fazendo as devidas modificações como por exemplo alterar a URL de envio para https://aydanomachado.com/mlclass/03_Validation.php.

**Atenção:** nessa atividade só será permitido **1 envio a cada 12h** pois o objetivo é fazer uma boa validação do modelo antes desse ser enviado.

Ainda nos mesmos moldes da Atividade 01 os arquivos `abalone_dataset.xlsx` ou `abalone_dataset.csv` devem ser utilizados para a construção e validação do modelo preditivo (classificador) e os arquivos `abalone_app.xlsx` ou `abalone_app.csv` utilizados para teste do modelo e as previsões enviadas para o servidor onde será registrado o desempenho do modelo construído, correspondente a sua acurácia.

## Descrição da base de dados

Esse conjunto de dados foi modificado a partir da base encontrada no [UCI Machine Learning Repository: Abalone Data Set](http://archive.ics.uci.edu/ml/datasets/Abalone).
Que foi originalmente utilizado no estudo Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994) "The Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip Abalone (H. rubra) from the North Coast and Islands of Bass Strait", Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288).

A base consiste de informações de um molusco chamado Abalone, e o objetivo do classificador é identificar o tipo do exemplar (entre as classes I, II e III) utilizando as informações fornecidas e detalhadas a seguir.

#### Abalone
(Origem: [Wikipédia, a enciclopédia livre](https://pt.wikipedia.org/wiki/Abalone))
"Haliotis (popularmente conhecidos em português e inglês por abalone, também em inglês por ear shell ou ormer, em espanhol por oreja de mar e abulone, em francês por oreille de mer, em italiano por abaloni e em alemão por seeohren) é um gênero de moluscos gastrópodes marinhos da família Haliotidae e o único gênero catalogado desta família. Foi proposto por Linnaeus em 1758 e contém diversas espécies em águas costeiras de quase todo o mundo. Na gastronomia, o abalone é um molusco valorizado em países asiáticos. Suas dimensões variam de dois a trinta centímetros."

#### Atributos do dataset:
1. **Sex**: M, F e I (infantil)
2. **Length**: maior medida em mm da concha
3. **Diameter**: diametro em mm perpendicular a medida Length
4. **Height**: altura em mm com a carne dentro da concha
5. **Whole weight**: peso em gramas de toda a abalone
6. **Shucked weight**: peso em gramas da carne
7. **Viscera weight**: peso em gramas das víceras após escorrer
8. **Shell weight**: peso em gramas para a concha após estar seca
9. **Type**: variável de classe (1, 2 ou 3) para o abalone

## Execução dos scripts Python

Dependências: `numpy`, `pandas`, `scikit-learn>=1.2` e `requests`.
Os caminhos padrão são relativos à pasta dos scripts. Exemplos a partir da raiz:

```bash
# Testes automatizados, sem envio real.
python -m unittest discover -s 03_Validation/tests -v

# Busca e compara KNN, Random Forest, regressão logística, HGB, SVM e Extra Trees.
python 03_Validation/localrun.py --n-iter 20 --cv 5

# Seleciona, avalia, reajusta e gera previsões sem enviar.
python 03_Validation/send_model.py --dry-run --n-iter 20 --cv 5

# Executa a mesma verificação e envia as previsões do vencedor.
python 03_Validation/send_model.py --n-iter 20 --cv 5
```

O fluxo automático de `send_model.py` é:

1. Reservar 20% do dataset para teste, antes das buscas.
2. Buscar parâmetros dos seis algoritmos nos 80% de treino, usando as mesmas
   partições estratificadas e escolhendo pela acurácia média na validação cruzada.
3. Selecionar o algoritmo/configuração com maior acurácia de CV. Empates exatos
   seguem a ordem KNN, RandomForest, LogReg, HGB, SVM, ExtraTrees.
4. Avaliar somente o vencedor no teste reservado e salvar `selection_report.json`
   com ranking, parâmetros e métricas. `--report` permite outro caminho.
5. Reajustar o pipeline vencedor em todo o dataset e gerar as previsões de aplicação.
6. Enviar, exceto quando `--dry-run` estiver presente. Erros de validação interrompem
   a execução antes do envio.

A acurácia de teste é informativa: não escolhe o vencedor nem funciona como um
limiar mínimo de aprovação. Não use esse teste para ajustes repetidos. A seleção
compara apenas as configurações avaliadas e não garante a maior acurácia no servidor.
O modo `--dry-run` executa inclusive o reajuste final; ao executar novamente sem
essa opção, a seleção é refeita. Com os mesmos dados e opções, a semente fixa
reproduz as partições e os candidatos. O relatório é sobrescrito a cada execução.

`--n-iter` limita candidatos por algoritmo; `--cv` controla as partições e `--jobs`
o paralelismo. Para conferir rapidamente: `--n-iter 2 --cv 3 --jobs 1`. Essa busca
curta serve para testar o fluxo, não substitui uma busca completa. `localrun.py`
aceita também `--nested`, que adiciona uma estimativa por CV aninhada apenas do KNN.

### Experimento específico de Random Forest

`randomforest.py` busca hiperparâmetros com validação cruzada no treino, avalia
no teste reservado e imprime acurácia, relatório por classe e matriz de confusão.
Salva os parâmetros em `forest_params.json`, sem enviar previsões ao servidor.

```bash
python 03_Validation/randomforest.py --n-iter 20 --cv 5 --jobs 2

# Compara a floresta com parâmetros fixos com os outros cinco modelos, sem enviar.
python 03_Validation/send_model.py --params 03_Validation/forest_params.json --dry-run
```

Esse experimento é opcional. `--params` fixa apenas a configuração candidata de
Random Forest; a seleção continua usando a acurácia de CV de todos os modelos.
`--params-output` permite escolher onde salvar os parâmetros da floresta.

### Pré-processamento compartilhado

`tratamento.py` é importado pelos scripts de treino e envio. Não precisa ser
executado separadamente e não exporta um CSV pré-processado. Preserva todas as
linhas e os atributos originais, valida as colunas e cria quatro razões:

- `bmi`: `whole_weight / height²` (nome usado no código).
- `length_dia_ratio`: `length / diameter`.
- `meat_yield`: `shucked_weight / whole_weight`.
- `shell_ratio`: `shell_weight / whole_weight`.

Os dois CSVs originais não contêm valores ausentes. No dataset rotulado, duas
observações têm `height = 0` (linhas 198 e 3080 do CSV, contando o cabeçalho).
A divisão inválida gera `NaN` em `bmi`, que o pipeline preenche com a mediana
aprendida no treino. A altura original permanece zero. No CSV de aplicação,
a criação das razões não gera ausências.

A categoria `sex` recebe codificação one-hot. A imputação pela moda está
configurada como proteção para eventuais ausências, mas não preenche nenhum
valor de `sex` nos dados atuais. Categorias desconhecidas são ignoradas pelo
codificador, sem interromper a previsão.

KNN, regressão logística e SVM usam `StandardScaler`; Random Forest, HGB e
Extra Trees não usam padronização. Imputação, categorias e escala são aprendidas
somente no treino de cada partição. O fluxo de seleção não aplica SMOTE,
winsorização ou PCA. A opção de excluir atributos originais foi removida.

### Exemplo básico de envio

`abalone_csv.py` permanece como exemplo alternativo simples com KNN k=3. Ele não faz
a seleção automática; para isso, use `send_model.py`. O endpoint e os campos
`dev_key` e `predictions` permanecem iguais. Confira `--dev-key` e respeite o limite
de um envio a cada 12h. Importar os módulos não treina nem envia previsões.

### Comparação dos modelos

A seleção automática inclui SVM com kernel RBF e Extra Trees. Ambos
comparam atributos originais com e sem as quatro razões. A opção vencedora fica
registrada como `prep__features__include_ratios` no relatório. A busca avalia seis algoritmos.
`--jobs 2` limita a quantidade de ajustes paralelos.

Uma acurácia maior na validação não garante melhora no servidor. O teste reservado
já foi consultado em experimentos anteriores: não deve orientar novos ajustes.


### Resultados registrados

Na execução com `--n-iter 20 --cv 5` e semente 42, a SVM com kernel RBF foi
selecionada com `C = 1.0`, `gamma = scale`, sem pesos de classe e com as quatro
razões. O experimento usou 2.505 amostras para busca e 627 para teste local.

- Acurácia média na validação cruzada: **65,83%**.
- Acurácia no teste local: **67,15%**; F1-macro: **67,06%**.
- Acurácia no servidor, informada pela equipe Delta: **66,12%**
  (`accuracy = 0.661244019138756`, `status = success`).
- Resultado anterior no servidor: **65,07%** (`old_accuracy = 0.65071770334928`).
  Ganho de **1,05 ponto percentual**.

A matriz de confusão reproduzida no teste local tem linhas de classe real e
colunas de classe prevista, ambas na ordem 1, 2 e 3:

```text
166   44    6
 34  116   51
 14   57  139
```

São 421 acertos em 627 amostras. Essa matriz é do teste local: o retorno do
servidor contém apenas a acurácia, sem rótulos ou matriz de confusão externa.
Os resultados acima descrevem a execução registrada, não qualquer nova busca
com outras opções. A diferença observada não demonstra superioridade estatística.
