library(e1071)
library(caret)
library(pROC)
library(doParallel)

raw<-read.csv("C:/Users/monis/Documents/MTech-KE/Sem2/CI1/CA/Thyroid/thyroid.csv",header=T,na.strings=c(""," ","NA"))

#splitting into training and testing data. Here,0.8 i.e. 80% is taken as training data
index <- sample(nrow(raw), 0.8 * nrow(raw))
trainingData <- raw[index, ]
testingData <- raw[-index, ]

#Baseline approach
svm_results1<-svm(as.factor(Functioning) ~.,data=trainingData)
svmprediction1<-predict(svm_results1,testingData)
confusion_matrix1 <- confusionMatrix(svmprediction1,testingData$Functioning)
print(confusion_matrix1)
confusion_matrix1$table
ROC_SVM1<-multiclass.roc(svmprediction1,testingData$Functioning)
print(ROC_SVM1)
auc1 <- ROC_SVM1['auc']
print(auc1)
#plot ROC for multiclass
rs1 <- ROC_SVM1[['rocs']]
plot(rs1[[1]], main = "ROC For Baseline SVM")
sapply(2:length(rs1),function(i) lines.roc(rs1[[i]],col=i))

#linear kernel
svm_results2<-svm(as.factor(Functioning) ~.,data=trainingData, kernel = "linear", cost = 1)
svmprediction2<-predict(svm_results2,testingData)
confusion_matrix2 <- confusionMatrix(svmprediction2,testingData$Functioning)
print(confusion_matrix2)
confusion_matrix2$table
ROC_SVM2<-multiclass.roc(svmprediction2,testingData$Functioning)
print(ROC_SVM2)
auc2 <- ROC_SVM2['auc']
print(auc2)
#plot ROC for multiclass
rs2 <- ROC_SVM2[['rocs']]
plot(rs2[[1]], main = "ROC For Linear Kernel SVM")
sapply(2:length(rs2),function(i) lines.roc(rs2[[i]],col=i))

#randomly changing cost and sigma
svm_results3<-svm(as.factor(Functioning) ~.,data=trainingData, kernel = "radial", cost = 2, sigma = 0.3)
svmprediction3<-predict(svm_results3,testingData)
confusion_matrix3 <- confusionMatrix(svmprediction3,testingData$Functioning)
print(confusion_matrix3)
confusion_matrix3$table
ROC_SVM3<-multiclass.roc(svmprediction3,testingData$Functioning)
print(ROC_SVM3)
auc3 <- ROC_SVM3['auc']
print(auc3)
#plot ROC for multiclass
rs3 <- ROC_SVM3[['rocs']]
plot(rs3[[1]], main = "ROC For Radial Kernel SVM")
sapply(2:length(rs3),function(i) lines.roc(rs3[[i]],col=i))

#For svm, train in caret does not accept numbers in target so change them to strings
modifiedData <- raw
for (row in 1:nrow(modifiedData)){
  if(modifiedData[row,'Functioning'] == 1)
    modifiedData[row,'Condition'] <- 'Normal' 
  else if(modifiedData[row,'Functioning'] == 2)
    modifiedData[row,'Condition'] <- 'Hyperthyroidism'
  else
    modifiedData[row,'Condition'] <- 'Hypothyroidism'
}

#remove functioning column and keep condition as target
modifiedData <- modifiedData[,-22]

index <- sample(nrow(modifiedData), 0.8 * nrow(raw))
trainingData1 <- modifiedData[index, ]
testingData1 <- modifiedData[-index, ]

#values for tuning sigma and c - radial kernel
svmGrid <- expand.grid(sigma= 2^c(-25, -20, -15,-10, -5, 0), C= 2^c(0:5))
set.seed(45)

cls = makeCluster(8)
registerDoParallel(cls)

train_ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, allowParallel = TRUE)
svm_cv <- train(Condition ~ ., data = trainingData1, method = "svmRadial", tuneGrid = svmGrid,
                metric = "Accuracy", trControl = train_ctrl)
results <- svm_cv$results
results$sigma2 <- paste0("2^", format(log2(results$sigma)))
best_tune <- svm_cv$bestTune
best_sigma <- best_tune$sigma
print(best_sigma)
best_C <- best_tune$C
print(best_C)

svm_results4<-svm(as.factor(Functioning) ~.,data=trainingData, kernel = "radial", cost = best_C, sigma = best_sigma)
svmprediction4<-predict(svm_results4,testingData)
confusion_matrix4 <- confusionMatrix(svmprediction4,testingData$Functioning)
print(confusion_matrix4)
confusion_matrix4$table
ROC_SVM4<-multiclass.roc(svmprediction4,testingData$Functioning)
print(ROC_SVM4)
auc4 <- ROC_SVM4['auc']
print(auc4)
#plot ROC for multiclass
rs4 <- ROC_SVM4[['rocs']]
plot(rs4[[1]], main = "ROC For Tuned SVM-Radial Kernel")
sapply(2:length(rs4),function(i) lines.roc(rs4[[i]],col=i))

#values for tuning c - linear kernel
svmGrid1 <- expand.grid(C= 2^c(0:5))
set.seed(45)

cls1 = makeCluster(8)
registerDoParallel(cls1)

train_ctrl1 <- trainControl(method = "cv", number = 10, classProbs = TRUE, allowParallel = TRUE)
svm_cv1 <- train(Condition ~ ., data = trainingData1, method = "svmLinear", tuneGrid = svmGrid1,
                 metric = "Accuracy", trControl = train_ctrl1)
results1 <- svm_cv1$results
best_tune1 <- svm_cv1$bestTune
best_C1 <- best_tune1$C
print(best_C1)

svm_results5<-svm(as.factor(Functioning) ~.,data=trainingData, kernel = "linear", cost = best_C1)
svmprediction5<-predict(svm_results5,testingData)
confusion_matrix5 <- confusionMatrix(svmprediction5,testingData$Functioning)
print(confusion_matrix5)
confusion_matrix5$table
ROC_SVM5<-multiclass.roc(svmprediction5,testingData$Functioning)
print(ROC_SVM5)
auc5 <- ROC_SVM5['auc']
print(auc5)
#plot ROC for multiclass
rs5 <- ROC_SVM5[['rocs']]
plot(rs5[[1]], main = "ROC For Tuned SVM-Linear Kernel")
sapply(2:length(rs5),function(i) lines.roc(rs5[[i]],col=i))

#After comparing the results, fine tuned svm model with radial kernel is found to be the best one as it gives 0.9771 accuracy
