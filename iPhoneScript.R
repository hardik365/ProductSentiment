library(doParallel)
library(caret)
library(readr)
library(ggplot2)
library(plotly)
library(corrplot)
library(e1071)
library(kknn)
set.seed(123)

# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6
# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(22)
# Register Cluster
registerDoParallel(cl)
# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() 

iphoneDF <-read.csv("iphone_smallmatrix_labeled_8d.csv")
plot_ly(iphoneDF, x= ~iphoneDF$iphonesentiment, type='histogram')%>%
  layout(title = "iPhone Sentiment Histogram")

options(max.print=1000000)

iphoneDF$iphonesentiment <- as.factor(iphoneDF$iphonesentiment)

summary(iphoneDF)
str(iphoneDF)
#no nulls

corrData <- cor(iphoneDF)
corrData
corrplot(corrData)
#no highly correlated attributes, but there are some zeros

##CorDataFrame
# create a new data set and remove features highly correlated with the dependent 
iphoneCOR <- iphoneDF[,!names(iphoneDF) %in% c("nokiacampos", "sonycamneg", "nokiacamneg","sonydisneg","nokiadisunc","sonyperpos", "nokiaperpos","sonyperneg","nokiaperneg","iphone", "ios", "iphonedispos", "iphonedisneg", "iphonedisunc", "iphoneperpos", "iphoneperneg","iphoneperunc","iosperpos","iosperneg", "iosperunc")]

##NZV Dataframa
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 
nzvMetrics <- nearZeroVar(iphoneDF, saveMetrics = TRUE)
nzvMetrics
# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(iphoneDF, saveMetrics = FALSE) 
nzv
# create a new data set and remove near zero variance features
iphoneNZV <- iphoneDF[,-nzv]
str(iphoneNZV)


##RFE
# Let's sample the data before using RFE
iphoneSample <- iphoneDF[sample(1:nrow(iphoneDF), 1000, replace=FALSE),]
# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)
# Get results
rfeResults
# Plot results
plot(rfeResults, type=c("g", "o"))
# create new data set with rfe recommended features
iphoneRFE <- iphoneDF[,predictors(rfeResults)]
# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphoneDF$iphonesentiment
# review outcome
str(iphoneRFE)

iphoneDF$iphonesentiment <- as.factor(iphoneDF$iphonesentiment)
iphoneCOR$iphonesentiment <- as.factor(iphoneCOR$iphonesentiment)
iphoneNZV$iphonesentiment <- as.factor(iphoneNZV$iphonesentiment)
iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)


#iphoneDF (original one) dataframe segmented
iphoneDFPart <- createDataPartition(iphoneDF$iphonesentiment, p = .70, list = FALSE)
iphoneDFTraining <- iphoneDF[iphoneDFPart,]
iphoneDFTesting <- iphoneDF[-iphoneDFPart,]
iphone


#iphoneCor dataframe segmented
iphoneCORPart <- createDataPartition(iphoneCOR$iphonesentiment, p = .70, list = FALSE)
iphoneCORTraining <- iphoneCOR[iphoneCORPart,]
iphoneCORTesting <- iphoneCOR[-iphoneCORPart,]

#iphoneNZV dataframe segmented
iphoneNZVPart <- createDataPartition(iphoneNZV$iphonesentiment, p = .70, list = FALSE)
iphoneNZVTraining <- iphoneNZV[iphoneNZVPart,]
iphoneNZVTesting <- iphoneNZV[-iphoneNZVPart,]

#iphoneRFE dataframe segmented
iphoneRFEPart <- createDataPartition(iphoneRFE$iphonesentiment, p = .70, list = FALSE)
iphoneRFETraining <- iphoneRFE[iphoneRFEPart,]
iphoneRFETesting <- iphoneRFE[-iphoneRFEPart,]

fitControlRF <- trainControl(method = "repeatedcv", number = 10, repeats = 5 ,search = 'random')


#lets try the whole data with RF
iphoneDFRF <- train(iphonesentiment~.,
               data = iphoneDFTraining, 
               method = "rf",
               metric="Accuracy",
               trControl = fitControlRF)
iphoneDFRF   
#plot((iphoneDFRF), main = "Random Forest Accuracy")
#mtry  Accuracy   Kappa    
#14    0.7758871  0.5657843
#29    0.7726716  0.5624581
#335    0.7701619  0.5591568


#lets try the COR data with RF
iphoneCorRF <- train(iphonesentiment~.,
                    data = iphoneCORTraining, 
                    method = "rf",
                    metric="Accuracy",
                    trControl = fitControlRF)
iphoneCorRF   
#mtry  Accuracy   Kappa    
#7    0.7051403  0.4110664
#29    0.7022561  0.4072145
#35    0.7011768  0.4059448

#lets try the NZV data with RF
iphoneNZVRF <- train(iphonesentiment~.,
                     data = iphoneNZVTraining, 
                     method = "rf",
                     metric="Accuracy",
                     trControl = fitControlRF)
iphoneNZVRF  
#  mtry  Accuracy   Kappa    
#1     0.6927692  0.3477917
#6     0.7551249  0.5223198
#7     0.7528350  0.5193517

#lets try the RFE data with RF
iphoneRFERF <- train(iphonesentiment~.,
                     data = iphoneRFETraining, 
                     method = "rf",
                     metric="Accuracy",
                     trControl = fitControlRF)
iphoneRFERF  
# mtry  Accuracy   Kappa    
#6    0.7756474  0.5662654
#14    0.7676095  0.5562885
#16    0.7665303  0.5549904

##Let's try to see how well c5 works
fitControlC5 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, returnResamp="all")
gridC5 <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )


iphoneDFx <- iphoneDFTraining[,!names(iphoneDFTraining) %in% c("iphonesentiment")]
iphoneDFy <- iphoneDFTraining$iphonesentiment

iphoneCORx <- iphoneCORTraining[,!names(iphoneCORTraining) %in% c("iphonesentiment")]
iphoneCORy <- iphoneCORTraining$iphonesentiment

iphoneNZVx <- iphoneNZVTraining[,!names(iphoneNZVTraining) %in% c("iphonesentiment")]
iphoneNZVy <- iphoneNZVTraining$iphonesentiment

iphoneRFEx <- iphoneRFETraining[,!names(iphoneRFETraining) %in% c("iphonesentiment")]
iphoneRFEy <- iphoneRFETraining$iphonesentiment


#iphoneDF with c5
iphoneDFc5<- train(x=iphoneDFx,y=iphoneDFy,tuneGrid=gridC5,trControl=fitControlC5,method="C5.0",verbose=FALSE)
iphoneDFc5
#  winnow  trials  Accuracy   Kappa    
#FALSE    1      0.7725279  0.5585876
#FALSE    5      0.7626689  0.5444190
#FALSE   10      0.7627689  0.5444276
#FALSE   15      0.7627689  0.5444276
#FALSE   20      0.7627689  0.5444276
#TRUE    1      0.7726587  0.5588903
#TRUE    5      0.7619906  0.5426158
#TRUE   10      0.7618363  0.5421687
#TRUE   15      0.7618363  0.5421687
#TRUE   20      0.7618363  0.5421687

#iphoneCor with c5
iphoneCORc5<- train(x=iphoneCORx,y=iphoneCORy,tuneGrid=gridC5,trControl=fitControlC5,method="C5.0",verbose=FALSE)
iphoneCORc5


#iphoneNVZ with c5
iphoneNZVc5<- train(x=iphoneNZVx,y=iphoneNZVy,tuneGrid=gridC5,trControl=fitControlC5,method="C5.0",verbose=FALSE)
iphoneNZVc5
#  winnow  trials  Accuracy   Kappa    
#FALSE    1      0.7568962  0.5233408

#iphoneRFE with c5
iphoneRFEc5<- train(x=iphoneRFEx,y=iphoneRFEy,tuneGrid=gridC5,trControl=fitControlC5,method="C5.0",verbose=FALSE)
iphoneRFEc5
#  winnow  trials  Accuracy   Kappa    
#FALSE    1      0.7718873  0.5572843

##Let's try to see how well SVM works


#Let's try SVM on DF
iphoneDFSVM <-svm(iphonesentiment ~ ., data=iphoneDFTraining)
iphoneDFPred <- predict(iphoneDFSVM,iphoneDFTesting)
confusionMatrix(iphoneDFPred,iphoneDFTesting$iphonesentiment)

#Let's try SVM on COR
iphoneCorSVM <-svm(iphonesentiment ~ ., data=iphoneCORTraining)
iphoneCorPred <- predict(iphoneCorSVM,iphoneCORTesting)
confusionMatrix(iphoneCorPred,iphoneCORTesting$iphonesentiment)

#Let's try SVM on NZV
iphoneNzvSVM <-svm(iphonesentiment ~ ., data=iphoneNZVTraining)
iphoneNzvPred <- predict(iphoneNzvSVM,iphoneNZVTesting)
confusionMatrix(iphoneNzvPred,iphoneNZVTesting$iphonesentiment)

#Let's try SVM on RFE
iphoneRFESVM <-svm(iphonesentiment ~ ., data=iphoneRFETraining)
iphoneRFEPred <- predict(iphoneRFESVM,iphoneRFETesting)
confusionMatrix(iphoneRFEPred,iphoneRFETesting$iphonesentiment)


##Let's try to see how well KKNN works
fitControlkknn <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


#Let's try kknn on DF
iphoneDFkknn <- train(iphonesentiment~.,
                    data = iphoneDFTraining, 
                    method = "kknn",
                    trControl = fitControlkknn)
iphoneDFkknn

#Let's try kknn on Cor
iphoneCORkknn <- train(iphonesentiment~.,
                      data = iphoneCORTraining, 
                      method = "kknn",
                      trControl = fitControlkknn)
iphoneCORkknn

#Let's try kknn on nzv
iphoneNZVkknn <- train(iphonesentiment~.,
                       data = iphoneNZVTraining, 
                       method = "kknn",
                       trControl = fitControlkknn)
iphoneNZVkknn

#Let's try kknn on rfe
iphoneRFEkknn <- train(iphonesentiment~.,
                       data = iphoneRFETraining, 
                       method = "kknn",
                       trControl = fitControlkknn)
iphoneRFEkknn




#Our data that we will do the prediction on
iphoneLargeMatrix  <-read.csv("iphoneLargeMatrix.csv")

#to validate our best model
validPred <- predict(iphoneDFRF,iphoneDFTesting)
confusionMatrix(validPred,iphoneDFTesting$iphonesentiment)
#It did just about as we expected!

#predict onto the data we wanted
finalPred <- predict(iphoneDFRF, iphoneLargeMatrix)
summary(finalPred)

iphoneLargeMatrix$iphonesentiment <- finalPred
write.csv(iphoneLargeMatrix, file="iphoneLargeMatrix.csv", row.names = TRUE)


plot_ly( x= ~finalPred, type='histogram')%>%
  layout(title = "iPhone Sentiment Histogram (Predicted)")

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)
