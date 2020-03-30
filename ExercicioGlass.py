# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


#carregando arquivo dos dados
vidro=pd.read_csv('glass.csv')
vidro.head()
len(vidro)
vidro.shape
vidro.columns
vidro['Type'].unique()

#previsores
previsores=vidro.iloc[:,0:9].values
previsores[:2]

#classe
classe=vidro.iloc[:,9].values
classe[:2]

#não precisa transforma dados categóricos em discretos (numéricos)
previsores[0]
classe[1]


#conjunto de treino e conjuunto de teste
#70%, 30%
previsores_treinamento,previsores_teste,classe_treinamento,classe_teste=train_test_split(previsores,classe,test_size=0.3,random_state=0)

len(previsores_treinamento)
len(previsores_teste)

#   knn

#objeto do tipo knn -> treinamento
#n_neighbors -> numero de vizinhos
#p -> 2, distância euclidiana ou p -> 1 é manhatan
knn=KNeighborsClassifier(n_neighbors=17,p=2) #nro ímpar de vizinhos para desempatar em caso de duas opções

#treinamento
knn.fit(previsores_treinamento,classe_treinamento)

#predict -> previsão dos dados
previsoes_knn=knn.predict(previsores_teste)
previsoes_knn

previsoes_knn[0]
classe_teste[0]

confusao_knn=confusion_matrix(previsoes_knn,classe_teste)
confusao_knn

score_knn=accuracy_score(previsoes_knn,classe_teste)
score_knn

#   naive

#criação do naive bayes
naive=GaussianNB()

#treinamento
naive.fit(previsores_treinamento,classe_treinamento)

#teste -> previsões
previsoes_naive=naive.predict(previsores_teste)
previsoes_naive

#validação
confusao_naive=confusion_matrix(previsoes_naive,classe_teste)
confusao_naive
score_naive=accuracy_score(previsoes_naive,classe_teste)
score_naive

#   svm

#SVC -> support vector machine
#treinamento
classificador=SVC(kernel='linear',random_state=1)
classificador.fit(previsores_treinamento,classe_treinamento)

#teste -> previsoes
previsoes_svm=classificador.predict(previsores_teste)
previsoes_svm

confusao_svm=confusion_matrix(previsoes_svm,classe_teste)
confusao_svm

score_svm=accuracy_score(previsoes_svm,classe_teste)
score_svm

#   árvore

#criação de árvore de decisão
#critério -> gini(grau de pureza) ou entropy(ganho de informação)
arvore=DecisionTreeClassifier(criterion="entropy")

#treinamento
arvore.fit(previsores_treinamento,classe_treinamento)

#conda install graphviz
from sklearn.tree import export_graphviz

export_graphviz(arvore,out_file='tree.dot',
                feature_names=vidro.columns[0:9],
                class_names=str(vidro['Type'].unique()),
                leaves_parallel=True,
                filled=True
                )

import pydot

(graph,)=pydot.graph_from_dot_file('tree.dot')
graph.write_png('arvore.png') #erro neste comando: FileNotFoundError: [WinError 2] "dot" not found in path.

#teste -> previsões
previsoes_arvore=arvore.predict(previsores_teste)
previsoes_arvore

#validação
confusao_arvore=confusion_matrix(previsoes_arvore,classe_teste)
confusao_arvore
score_arvore=accuracy_score(previsoes_arvore,classe_teste)
score_arvore

#   floresta

#RandomForest
#n_estimators -> número de árvores
floresta=RandomForestClassifier(n_estimators=100,
                                criterion="entropy")

#treinamento
floresta.fit(previsores_treinamento,classe_treinamento)

#teste
previsoes_floresta=floresta.predict(previsores_teste)
previsoes_floresta

#validação_floresta
confusao_floresta=confusion_matrix(previsoes_floresta,classe_teste)
confusao_floresta
score_floresta=accuracy_score(previsoes_floresta,classe_teste)
score_floresta

#resultado geral
resultado=pd.DataFrame([
        ['KNN',score_knn],
        ['Naive Bayes',score_naive],
        ['SVM',score_svm],
        ['Decision Tree',score_arvore],
        ['Random Forest',score_floresta]
        ],columns=['Método','Percentual'])
resultado
