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

galaxyDF <-read.csv("galaxy_smallmatrix_labeled_9d.csv")
plot_ly(galaxyDF, x= ~galaxyDF$galaxysentiment, type='histogram')%>%
  layout(title = "Galaxy Sentiment Histogram")

options(max.print=1000000)
summary(galaxyDF)
str(galaxyDF)
#no nulls

#we will exclude the others df for the galaxy data 
#as we feel it is not needed and redundant and we think the out of the box one all we need

galaxyDF$galaxysentiment <- as.factor(galaxyDF$galaxysentiment)

#galaxyDF (original one) dataframe segmented
galaxyDFPart <- createDataPartition(galaxyDF$galaxysentiment, p = .70, list = FALSE)
galaxyDFTraining <- galaxyDF[galaxyDFPart,]
galaxyDFTesting <- galaxyDF[-galaxyDFPart,]

fitControlG <- trainControl(method = "repeatedcv", number = 10, repeats = 5 ,search = 'random')

#lets try the whole data with RF
galaxyDFRF <- train(galaxysentiment~.,
                    data = galaxyDFTraining, 
                    method = "rf",
                    metric="Accuracy",
                    trControl = fitControlG)
galaxyDFRF  

##Let's try to see how well c5 works
fitControlC5 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, returnResamp="all")
gridC5 <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )

galaxyDFx <- galaxyDFTraining[,!names(galaxyDFTraining) %in% c("galaxysentiment")]
galaxyDFy <- galaxyDFTraining$galaxysentiment

#df with c5
galaxyDFc5<- train(x=galaxyDFx,y=galaxyDFy,tuneGrid=gridC5,trControl=fitControlC5,method="C5.0",verbose=FALSE)
galaxyDFc5

##lets try too see how svm works

#df with svm
galaxyDFSVM <-svm(galaxysentiment ~ ., data=galaxyDFTraining)
galaxyDFPred <- predict(galaxyDFSVM,galaxyDFTesting)
confusionMatrix(galaxyDFPred,galaxyDFTesting$galaxysentiment)

##Lets try to see how kknn does
fitControlkknn <- trainControl(method = "repeatedcv", number = 10, repeats = 5)        

#df with kknn
galaxyDFkknn <- train(galaxysentiment~.,
                      data = galaxyDFTraining, 
                      method = "kknn",
                      trControl = fitControlkknn)
galaxyDFkknn


#Our data that we will do the prediction on
galaxyLargeMatrix  <-read.csv("galaxyLargeMatrix.csv")

#to validate our best model
validPred <- predict(galaxyDFRF,galaxyDFTesting)
confusionMatrix(validPred,galaxyDFTesting$galaxysentiment)
#It did just about as we expected!


finalPredG <- predict(galaxyDFRF, galaxyLargeMatrix)
summary(finalPredG)

galaxyLargeMatrix$galaxysentiment <- finalPredG
write.csv(galaxyLargeMatrix, file="galaxyLargeMatrix.csv", row.names = TRUE)

plot_ly( x= ~finalPredG, type='histogram')%>%
  layout(title = "Galaxy Sentiment Histogram (Predicted)")

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)









