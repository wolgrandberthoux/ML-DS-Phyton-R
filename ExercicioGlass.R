#exercicio

#importando csv
vidro=read.table('glass.csv',head=T,sep=',',dec='.')
vidro
class(vidro)

#separando massa de treino e massa de teste
exemplo=sample(2,dim(vidro),replace=T,prob=c(0.7,0.3))
exemplo

vidro_treino=vidro[exemplo==1,]
vidro_teste=vidro[exemplo==2,]

#knn
library(class)
previsoes_knn=knn(vidro_treino[,1:9],vidro_teste[,1:9],vidro_treino[,10],k=15)
previsoes_knn
length(previsoes_knn)

#matriz de confusão
confusao_knn=table(vidro_teste[,10],previsoes_knn)
confusao_knn

resultado_knn=sum(diag(confusao_knn))/sum(confusao_knn)
resultado_knn #percentual

#naive
library(e1071) #biblioteca para treinamentos supervisionados

head(vidro)
vidro[,10]=factor(vidro[,10]) #convertendo coluna Type de integer para factor
levels(vidro[,10])
levels(vidro$Type)

#criação do modelo
vidro_treino[,10]=factor(vidro_treino[,10]) #convertendo coluna Type de integer para factor
modelo_naive=naiveBayes(Type~.,vidro_treino)
modelo_naive

#validação do modelo
vidro_teste[,10]=factor(vidro_teste[,10]) #convertendo coluna Type de integer para factor
previsoes_naive=predict(modelo_naive,vidro_teste)
previsoes_naive

previsoes_naive[1]
vidro_teste$Type[1]

#matriz de confusão (a diagonal principal mostra os acertos e secundária erros)
confusao_naive=table(vidro_teste$Type,previsoes_naive)
confusao_naive

#calculo do percentual
resultado_naive=(confusao_naive[1]+confusao_naive[4])/sum(confusao_naive)
resultado_naive #percentual

#svm
modelo_svm=svm(Type~.,vidro_treino)

previsoes_svm=predict(modelo_svm,vidro_teste)
previsoes_svm

confusao_svm=table(vidro_teste$Type,previsoes_svm)
confusao_svm

resultado_svm=(confusao_svm[1]+confusao_svm[4])/sum(confusao_svm)
resultado_svm #percentual

#árvore de decisão
library(rpart)

arvore=rpart(Type~.,vidro_treino,method="class")
arvore
plot(arvore)
text(arvore,use.n=T,all=T,cex=0.8)

previsoes_arvore=predict(arvore,newdata=vidro_teste)
previsoes_arvore

confusao_arvore=table(vidro_teste$Type,max.col(previsoes_arvore))
confusao_arvore

resultado_arvore=(confusao_arvore[1]+confusao_arvore[4])/sum(confusao_arvore)
resultado_arvore #percentual

#floresta aleatória
library(randomForest)

#floresta <- treinamento
floresta <- randomForest(Type~.,data=vidro_treino,ntree=100,importance=T)
floresta

#previsão -> validação
previsoes_floresta <- predict(floresta,vidro_teste)
previsoes_floresta

confusao_floresta <- table(vidro_teste$Type,previsoes_floresta)
confusao_floresta

resultado_floresta <- sum(confusao_floresta[1]+confusao_floresta[4])/sum(confusao_floresta)
resultado_floresta #percentual

#resultado geral
resultado=data.frame(c('KNN','Naive Bayes','SVM','Decision Tree','Random Forest'),
                     c(resultado_knn,resultado_naive,resultado_svm,resultado_arvore,resultado_floresta)
                     )
colnames(resultado)=c('Metodo','Percentual')
resultado
View(resultado)
