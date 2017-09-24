#Load libraries
library("caret")
#Download the data
if(!file.exists("pml-training.csv")){download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")}

if(!file.exists("pml-testing.csv")){download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")}

#Read the training data and replace empty values by NA
trainingDataSet<- read.csv("pml-training.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
testingDataSet<- read.csv("pml-testing.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
dim(trainingDataSet)

#[1] 19622   160

dim(testingDataSet)

#[1]  20 160
#Our data consists of 19622 values of 160 variables.

#Clean the data
#We remove columns with missing value.

trainingDataSet <- trainingDataSet[,(colSums(is.na(trainingDataSet)) == 0)]
dim(trainingDataSet)
## [1] 19622    60

testingDataSet <- testingDataSet[,(colSums(is.na(testingDataSet)) == 0)]
dim(testingDataSet)

#[1] 20 60

#We reduced our data to 60 variables.

#Preprocess the data

numericalsIdx <- which(lapply(trainingDataSet, class) %in% "numeric")

preprocessModel <-preProcess(trainingDataSet[,numericalsIdx],method=c('knnImpute', 'center', 'scale'))
pre_trainingDataSet <- predict(preprocessModel, trainingDataSet[,numericalsIdx])
pre_trainingDataSet$classe <- trainingDataSet$classe

pre_testingDataSet <-predict(preprocessModel,testingDataSet[,numericalsIdx])

#Removing the non zero variables
#Removing the variables with values near zero, that means that they have not so much meaning in the predictions

nzv <- nearZeroVar(pre_trainingDataSet,saveMetrics=TRUE)
pre_trainingDataSet <- pre_trainingDataSet[,nzv$nzv==FALSE]

nzv <- nearZeroVar(pre_testingDataSet,saveMetrics=TRUE)
pre_testingDataSet <- pre_testingDataSet[,nzv$nzv==FALSE]

#Validation set
#We want a 75% observation training dataset to train our model. We will then validate it on the last 70%.

set.seed(12031987)
idxTrain<- createDataPartition(pre_trainingDataSet$classe, p=3/4, list=FALSE)
training<- pre_trainingDataSet[idxTrain, ]
validation <- pre_trainingDataSet[-idxTrain, ]
dim(training) ; dim(validation)

#[1] 14718    28
#[1] 4904   28

#Train Model
#We train a model using random forest with a cross validation of 5 folds to avoid overfitting.

library(randomForest)
modFitrf <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE, importance=TRUE )
modFitrf

#Random Forest 

#14718 samples
#27 predictor
#5 classes: 'A', 'B', 'C', 'D', 'E' 

#No pre-processing
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 13246, 13247, 13245, 13246, 13246, 13247, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2    0.9927302  0.9908040
#14    0.9923222  0.9902882
#27    0.9882454  0.9851323

#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was mtry = 2. 
---------------------
#Interpretation
#Let’s plot the importance of each individual variable

 varImpPlot(modFitrf$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, cex = 0.6, main = "Importance of the Individual Principal Components")
#This plot shows each of the principal components in order from most important to least important.
-------------------
  
#Cross Validation Testing and Out-of-Sample Error Estimate
#Let’s apply our training model on our testing database, to check its accuracy.

#Accuracy and Estimated out of sample error

predValidRF <- predict(modFitrf, validation)

confus <- confusionMatrix(predValidRF, validation$classe)
confus
Confusion Matrix and Statistics

#          Reference
#Prediction    A    B    C    D    E
#         A 1391    2    0    0    2
#         B    2  946    1    0    0
#         C    0    2  852    1    0
#         D    0    0    4  800    0
#         E    0    0    0    3  898

#Overall Statistics
                                         
#               Accuracy : 0.9965         
#                 95% CI : (0.9945, 0.998)
#    No Information Rate : 0.2841         
#    P-Value [Acc > NIR] : < 2.2e-16      
                                         
#                  Kappa : 0.9956         
# Mcnemar's Test P-Value : NA             

#Statistics by Class:

#                     Class: A Class: B Class: C Class: D Class: E
#Sensitivity            0.9986   0.9958   0.9942   0.9950   0.9978
#Specificity            0.9989   0.9992   0.9993   0.9990   0.9993
#Pos Pred Value         0.9971   0.9968   0.9965   0.9950   0.9967
#Neg Pred Value         0.9994   0.9990   0.9988   0.9990   0.9995
#Prevalence             0.2841   0.1937   0.1748   0.1639   0.1835
#Detection Rate         0.2836   0.1929   0.1737   0.1631   0.1831
#Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
#Balanced Accuracy      0.9987   0.9975   0.9967   0.9970   0.9985

modFitrf
#Random Forest 

#14718 samples
#   27 predictor
#    5 classes: 'A', 'B', 'C', 'D', 'E' 

#No pre-processing
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 13246, 13247, 13245, 13246, 13246, 13247, ... 
#Resampling results across tuning parameters:

#  mtry  Accuracy   Kappa    
#   2    0.9927302  0.9908040
#  14    0.9923222  0.9902882
#  27    0.9882454  0.9851323

#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was mtry = 2. 

#We see there are very few variables out of this model.

accur <- postResample(validation$classe, predValidRF)
modAccuracy <- accur[[1]]
modAccuracy

#[1] 0.9965334

out_of_sample_error <- 1 - modAccuracy
out_of_sample_error

#[1] 0.003466558

#The estimated accuracy of the model is 99.7% and the estimated out-of-sample error based on our fitted model applied to the cross validation dataset is 0.3%.
----------------
  
#Application of this model on the 20 test cases provided
#We have already clean the test data base (teData). We delete the “problem id” column as it is useless for our analysis.

pred_final <- predict(modFitrf, pre_testingDataSet)
pred_final


#[1] B A B A A E D B A A B C B A E E A B B B
#Levels: A B C D E

#Here are our results, we will use them for the submission of this course project in the coursera platform.
