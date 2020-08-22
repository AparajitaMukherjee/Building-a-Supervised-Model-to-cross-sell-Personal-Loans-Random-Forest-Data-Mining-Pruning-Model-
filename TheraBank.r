getwd()
setwd("C:/Users/HP/Documents/R Dataset")
TheraBank <- read.csv("GL_Thera_Bank_Personal_Loan_Modelling.csv", header = T)
names(TheraBank)

#remove ID
TheraBank <- TheraBank[,-1]
names(TheraBank)

#EDA
summary(TheraBank)
str(TheraBank)
library(psych)
describe(TheraBank)
dim(TheraBank)

#Check for Missing Values
library(DataExplorer)
plot_missing(TheraBank)

#remove rows/records with missing values
TheraBank <- na.omit(TheraBank)
plot_missing(TheraBank)

#change pincode to character
TheraBank$ZIPCode = as.character(TheraBank$ZIPCode)
str(TheraBank)

#convert target integer valiables into factor variables
TheraBank$PersonalLoan  = ifelse(TheraBank$PersonalLoan==1,1,0)
TheraBank$PersonalLoan  = as.factor(TheraBank$PersonalLoan)
str(TheraBank)
View(TheraBank)

#Check Correlation between the independent variables
names(TheraBank)
cormat <- round(cor(TheraBank[,c(1,2,3,5,6,7,8,10,11,12,13)]),2)
cor.plot(cormat)
plot_correlation(TheraBank[,c(1,2,3,5,6,7,8,10,11,12,13)])

#Frequency distribution of categorical variable
table(TheraBank[,c(4)])

plot_histogram(TheraBank)
plot_boxplot(TheraBank, by = TheraBank$PersonalLoan)

#CLUSTERING
names(TheraBank)
TheraClust <- TheraBank[-c(9)]
library(NbClust)
library(factoextra)
fviz_nbclust(TheraClust, kmeans, method = "wss")+
  geom_vline(xintercept = 2, linetype = 2)+
  labs(subtitle = "Elbow Method")

#applying k-means cluster
kmeans.cluster = kmeans(TheraClust, 2)
kmeans.cluster$size
kmeans.cluster$cluster
kmeans.cluster$withinss

#plotting the k-means
fviz_cluster(kmeans.cluster, data = TheraClust[-c(4)])


#Splitting the data(70-30) into train and test datasets
library(caTools)
set.seed(100)
TheraBank_Split <- sample(1:nrow(TheraBank),0.70*nrow(TheraBank))
CARTtrain <- TheraBank[TheraBank_Split,]
CARTtest <- TheraBank[-TheraBank_Split,]
dim(CARTtrain)
dim(CARTtest)  

#Checking the balance in the dataset
names(CARTtrain)
table(CARTtrain$PersonalLoan)

## Calculate the response rate
sum(CARTtrain$PersonalLoan == "1")/nrow(CARTtrain)
sum(CARTtest$PersonalLoan == "1")/nrow(CARTtest)


#Build CART model
library(rpart)
library(rpart.plot)  
treetrain <- rpart(formula = PersonalLoan ~ ., data=CARTtrain, method="class", minbucket = 10, cp=0.03560)
treetest <- rpart(formula = PersonalLoan ~ ., data=CARTtest, method="class", minbucket = 10, cp=0.0548)
rpart.plot(treetrain)  
rpart.plot(treetest)
printcp(treetrain)
printcp(treetest)  
plotcp(treetrain)  
plotcp(treetest)

#Pruning the tree - in order to avoid overfitting
ptreetrain <- prune(treetrain,cp=0.03560,"CP")
ptreetest <- prune(treetest,cp=0.0548,"CP")

##check the updated tree
##plot tree
rpart.plot(ptreetrain)
rpart.plot(ptreetest)

##print cp value
printcp(ptreetrain)
printcp(ptreetest)

##Use this tree to do the prediction on train as well as test data set
CARTtrain$CART.Pred = predict(ptreetrain,data=CARTtrain,type="class")
CARTtrain$CART.Score = predict(ptreetrain,data=CARTtrain,type="prob")[,"1"]
CARTtrain$CART.Pred=ifelse((CARTtrain$CART.Score>0.8),1,0)
View(CARTtrain)
CARTtest$CART.Pred = predict(ptreetest,CARTtest,type="class")
CARTtest$CART.Score = predict(ptreetest,CARTtest,type="prob")[,"1"]
CARTtest$CART.Pred=ifelse((CARTtest$CART.Score>0.8),1,0)
View(CARTtest)


table(CARTtrain$PersonalLoan)
3164/(3164+323)
table(CARTtest$PersonalLoan)
1340/(1340+155)


#plot the confusion matrix
library(ggplot2)
library(rlang)
library(caret)

table(CARTtrain$CART.Pred,CARTtrain$PersonalLoan)
table(CARTtest$CART.Pred,CARTtest$PersonalLoan)
CARTtrain$CART.Pred  = ifelse(CARTtrain$CART.Pred==1,1,0)
CARTtrain$CART.Pred  = as.factor(CARTtrain$CART.Pred)
CARTtest$CART.Pred  = ifelse(CARTtest$CART.Pred==1,1,0)
CARTtest$CART.Pred  = as.factor(CARTtest$CART.Pred)
str(CARTtrain)
str(CARTtest)
confusionMatrix(CARTtrain$CART.Pred,CARTtrain$PersonalLoan)
confusionMatrix(CARTtest$CART.Pred,CARTtest$PersonalLoan)



##Buildig Random Forest model

library(randomForest)
set.seed(100)
TheraBank_Index <- sample(1:nrow(TheraBank),0.70*nrow(TheraBank))
RFtrain <- TheraBank[TheraBank_Index,]
RFtest <- TheraBank[-TheraBank_Index,]

##import randomForest library for building random forest model
?tuneRF
names(RFtrain)
x=RFtrain[,-c(9)]
y=RFtrain$PersonalLoan
bestmtry=tuneRF(x,y,stepFactor = 1.5,improve = 1e-5,ntreeTry =201)
bestmtry <- sqrt(13)
print(bestmtry)


##Build the first RF model
Rforest = randomForest(PersonalLoan~.,data=RFtrain,ntree=101,mtry=4,nodesize=10,importance=TRUE)

##Print the model to see the OOB and error rate
print(Rforest)

plot(Rforest,main="")
legend("topright",c("OOB","0","1"),text.col = 1:6,lty = 1:3,col=1:3)
title(main="Error Rates random Forest")

##Identify the importance of the variables
importance(Rforest)
varImpPlot(Rforest)


##Tune up the RF model to find out the best mtry
set.seed(1000)
tRforest = tuneRF(x=RFtrain[,-c(9)],y=RFtrain$PersonalLoan,mtrystart = 3,stepfactor=1.5,ntree=101,improve=0.0001,
                  nodesize=10,trace=TRUE,plot=TRUE,doBest=TRUE,importance=TRUE)
#pick mtry value where oob error is least

##Build the refined RF model
Rforest = randomForest(PersonalLoan~.,data=RFtrain,ntree=101,mtry=4,nodesize=10,importance=TRUE)
print(Rforest)
plot(Rforest)


##Use this tree to do the prediction on train as well as test data set
RFtrain$RF.Pred = predict(Rforest,data=RFtrain,type="class")
RFtrain$RF.Score = predict(Rforest,data=RFtrain,type="prob")[,"1"]
RFtrain$RF.Pred=ifelse((RFtrain$RF.Score>0.8),1,0)
View(RFtrain)
RFtest$RF.Pred = predict(Rforest,RFtest,type="class")
RFtest$RF.Score = predict(Rforest,RFtest,type="prob")[,"1"]
RFtest$RF.Pred=ifelse((RFtest$RF.Score>0.8),1,0)
table(RFtrain$RF.Pred,RFtrain$PersonalLoan)

RFtrain$RF.Pred  = ifelse(RFtrain$RF.Pred==1,1,0)
RFtrain$RF.Pred  = as.factor(RFtrain$RF.Pred)
RFtest$RF.Pred  = ifelse(RFtest$RF.Pred==1,1,0)
RFtest$RF.Pred  = as.factor(RFtest$RF.Pred)
confusionMatrix(RFtrain$RF.Pred,RFtrain$PersonalLoan)
confusionMatrix(RFtest$RF.Pred,RFtest$PersonalLoan)


#MODEL PERFORMANCE MEASURES

library(ROCR)
library(ineq)
library(InformationValue)

##Confusion Metrix
## CART Model
CART_CM_train = table(CARTtrain$PersonalLoan,CARTtrain$CART.Pred)
CART_CM_test = table(CARTtest$PersonalLoan,CARTtest$CART.Pred)
CART_CM_train
CART_CM_test

## RF Model Confusion Metrix
RF_CM_train = table(RFtrain$PersonalLoan,RFtrain$RF.Pred)
RF_CM_test = table(RFtest$PersonalLoan,RFtest$RF.Pred)
RF_CM_train
RF_CM_test

## Error Rate
(CART_CM_train[1,2]+CART_CM_train[2,1])/nrow(CARTtrain) #1,2; 2,1 refers to the placements
(CART_CM_test[1,2]+CART_CM_test[2,1])/nrow(CARTtest)

## Error Rate
(RF_CM_train[1,2]+RF_CM_train[2,1])/nrow(RFtrain)
(RF_CM_test[1,2]+RF_CM_test[2,1])/nrow(RFtest)


##Accuracy
(CART_CM_train[1,1]+CART_CM_train[2,2])/nrow(CARTtrain)
(CART_CM_test[1,1]+CART_CM_test[2,2])/nrow(CARTtest)

##Accuracy
(RF_CM_train[1,1]+RF_CM_train[2,2])/nrow(RFtrain)
(RF_CM_test[1,1]+RF_CM_test[2,2])/nrow(RFtest)


## CART MODEL - ROC Curve
#train
CARTpredobjtrain = prediction(CARTtrain$CART.Score, CARTtrain$PersonalLoan)
CARTpreftrain = performance(CARTpredobjtrain,"tpr","fpr")
plot(CARTpreftrain)
#test
CARTpredobjtest = prediction(CARTtest$CART.Score, CARTtest$PersonalLoan)
CARTpreftest = performance(CARTpredobjtest,"tpr","fpr")
plot(CARTpreftest)


## RF MODEL - ROC Curve
##train
RFpredobjtrain = prediction(RFtrain$RF.Score,RFtrain$PersonalLoan)
RFpreftrain = performance(RFpredobjtrain,"tpr","fpr")
plot(RFpreftrain)
#test
RFpredobjtest = prediction(RFtest$RF.Score,RFtest$PersonalLoan)
RFpreftest = performance(RFpredobjtest,"tpr","fpr")
plot(RFpreftest)


## CART MODEL
##KS
max(CARTpreftrain@y.values[[1]]-CARTpreftrain@x.values[[1]])
max(CARTpreftest@y.values[[1]]-CARTpreftest@x.values[[1]])

## RF MODEL
##KS
max(RFpreftrain@y.values[[1]]-RFpreftrain@x.values[[1]])
max(RFpreftest@y.values[[1]]-RFpreftest@x.values[[1]])


## CART MODEL
##AUC
CARTauctrain=performance(CARTpredobjtrain,"auc")
as.numeric(CARTauctrain@y.values)
CARTauctest=performance(CARTpredobjtest,"auc")
as.numeric(CARTauctest@y.values)

## RF MODEL
##AUC
RFauctrain=performance(RFpredobjtrain,"auc")
as.numeric(RFauctrain@y.values)
RFauctest=performance(RFpredobjtest,"auc")
as.numeric(RFauctest@y.values)


## CART MODEL
##gini
ineq(CARTtrain$CART.Score,"gini")
ineq(CARTtest$CART.Score,"gini")

## RF MODEL
##gini
ineq(RFtrain$RF.Score,"gini")
ineq(RFtest$RF.Score,"gini")


## CART MODEL
##Concordance
Concordance(actuals=CARTtrain$PersonalLoan, predictedScores = CARTtrain$CART.Score)
Concordance(actuals=CARTtest$PersonalLoan, predictedScores = CARTtest$CART.Score)

## RF MODEL
##Concordance
Concordance(actuals=RFtrain$PersonalLoan,predictedScores = RFtrain$RF.Score)
Concordance(actuals=RFtest$PersonalLoan,predictedScores = RFtest$RF.Score)