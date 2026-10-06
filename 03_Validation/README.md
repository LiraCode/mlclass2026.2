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

`randomforest.py` disponível para um experimento exclusivo de Random Forest
que salva `forest_params.json`. Esse passo não é obrigatório para o envio automático.
Opcionalmente, `send_model.py --params 03_Validation/forest_params.json` avalia essa
configuração fixa de floresta na comparação com os demais algoritmos; ela não
força a seleção da floresta.

`tratamento.py` concentra os pipelines: preserva as linhas e os atributos originais,
adiciona quatro razões e aprende imputação, categorias e escala apenas nos treinos.
Árvores dispensam escala. SMOTE, winsorização e PCA não são aplicados por padrão.
Executá-lo diretamente exporta atributos determinísticos; não é um pré-requisito.

`abalone_csv.py` permanece como exemplo alternativo simples com KNN k=3. Ele não faz
a seleção automática; para isso, use `send_model.py`. O endpoint e os campos
`dev_key` e `predictions` permanecem iguais. Confira `--dev-key` e respeite o limite
de um envio a cada 12h. Importar os módulos não treina nem envia previsões.

### Ampliação da comparação

SVM com kernel RBF e Extra Trees foram acrescentados à seleção automática. Ambos
comparam atributos originais com e sem as quatro razões. A opção vencedora fica
registrada como `prep__features__include_ratios` no relatório. Os espaços de busca
dos quatro classificadores anteriores foram preservados, permitindo comparar a
ampliação com a configuração anterior usando a mesma semente e número de partições.
A busca agora custa mais tempo porque avalia seis algoritmos. `--jobs 2` limita a
quantidade de ajustes paralelos. Nenhum novo pacote é necessário.

Uma acurácia maior na validação não garante melhora no servidor. O teste reservado
já foi consultado em experimentos anteriores: não deve orientar novos ajustes.
