# Adaboost


## Documentação:


### I - Organização:

A implementação do Adaboost está localizada em ./classifier/my_adaboost.py e o notebook com os testes de performace estão em ./classifier-analysis.ipynb.

---
### II - Execução deste trabalho:

Instalando as dependências por meio do ambiente virtual:

~~~
$ python3 -m venv venv
$ venv/bin/pip3 install -r requirements.txt 
~~~

Depois da intalação, basta excutar célula por célula do notebook que contém os testes de performace.

---
### III -  Implementação do MyAdaboostClassifier:

O MyAdaboostClassifier recebe dois valores na inicialização da classe, o número de estimadores (n_estimators) e o tipo de classificador fraco do qual os estimadores são constituídos (classifier_model), e conta com 3 métodos públicos (fit, predict, score).

* ### .fit

    O método fit recebe como argumento uma lista de features númericas e uma lista de labels binárias (-1 ou 1) na qual os classificadores serão treinados. Nesse método, treinamos os n classificadores definidos na inicialização da classe da seguinte maneira: 
    
    - a cada classificador que treinamos calculamos para ele o o seu respectivo peso de decisão $\alpha$:

        $$
        \alpha = \frac{1}{2} ln\Big(\frac{a}{1 - a}\Big)
        $$
    
        onde $a$ é a taxa de acertos simples do classificador;

    - em seguida, ajustamos os pesos de cada um dos casos de treino ($w_i$) que serão utilizados pelo próximo classificador de acordo com que a acertividade do atual classificador, da seguinte maneira:
    
        $$
        w_i \leftarrow \frac{w_i}{Z} * e^{(-1)^b \alpha }
        $$

    onde $b$ é um booleano, que é 1 caso o atual classificador tenha acertado determinado caso de teste e 0 caso não, e $Z$ é uma constante normalizadora.

    O primeiro classificador, diferentemente dos demais, tem um peso padrão para todos os casos de treino que é $1/m$, onde $m$ é o tamanho da lista dos casos de treino. 

* ### .predict

    Esse método recebe como argumento uma lista de features e retorna para cada uma delas uma label fruto do processo de aprendizagem do MyAdaboost.

    A label retornada para cada caso de teste é dado pelo sinal da combinação linear das labels oriundas da predição dos n classificadores fracos multiplicadas pelo peso de decisão de cada um deles. Ou seja, a categoria ($l$) retornada pelo predict é dada por:

    $$
    l = sign\Big(\sum_{i=1}^n \alpha_i e_i\Big)
    $$

    onde $\alpha_i$ é o peso da decisão do estimador $e_i$.


* ### .score

    Essa funcão recebe uma lista de features e uma lista de labels de casos de teste e retorna um float referente a taxa de acertos da predição do MyAdaboost. A lista de features é usada no .predict e seu retorno é comparado cm a lista de features esperadas.

---
### IV - Observações:

* As analises da performace do classificador e suas respectivas interpretações estão no notebook ./classifier-analysis.ipynb;
* Disponível no repositório [Adaboost](https://github.com/g-bordoni/Adaboost)




