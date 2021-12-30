library(MASS)
library(caret)
library(ggplot2)
library(nnet)
library(glmnet)
library(gbm)
library(Metrics)
library(randomForest)
library(corrplot)
library(obliqueRF)
library(e1071)
library(quantregForest)
library(inTrees)
library(plyr)


# Load Dataset
obesity_data <- read.table("F:\\Predictive modeling\\CourseProject\\ObesityDataSet_raw_and_data_sinthetic.csv", header = TRUE, sep = "," )
summary(obesity_data)
View(obesity_data)
names(obesity_data)

# Encode categorical variables
obesity_data$Gender <- factor(obesity_data$Gender, 
                              levels = c("Male","Female"), 
                              labels = c(1,2))

obesity_data$family_history_with_overweight <- factor(obesity_data$family_history_with_overweight, 
                                                      levels = c("yes","no"), 
                                                      labels = c(1,2))

obesity_data$FAVC <- factor(obesity_data$FAVC, 
                            levels = c("yes","no"), 
                            labels = c(1,2))

obesity_data$CAEC <- factor(obesity_data$CAEC, 
                            levels = c("Sometimes","Frequently","Always"), 
                            labels = c(1,2,3))

obesity_data$SMOKE <- factor(obesity_data$SMOKE, 
                             levels = c("yes","no"), 
                             labels = c(1,2))

obesity_data$SCC <- factor(obesity_data$SCC, 
                           levels = c("yes","no"), 
                           labels = c(1,2))

obesity_data$CALC <- factor(obesity_data$CALC, 
                            levels = c("Sometimes","Frequently","no"), 
                            labels = c(1,2,3))

obesity_data$MTRANS <- factor(obesity_data$MTRANS, 
                              levels = c("Public_Transportation","Walking","Automobile","Bike","Motorbike"), 
                              labels = c(1,2,3,4,5))

obesity_data$NObeyesdad <- factor(obesity_data$NObeyesdad)

# Change the column name of the class variable
colnames(obesity_data)[colnames(obesity_data) == 'NObeyesdad'] <- 'class'
colnames(obesity_data)[colnames(obesity_data) == 'family_history_with_overweight'] <- 'fm_history'

# Convert numeric variables
obesity_data$Age <- as.numeric(obesity_data$Age)
obesity_data$Height <- as.numeric(obesity_data$Height)
obesity_data$Weight <- as.numeric(obesity_data$Weight)

# Convert float to intergers
obesity_data$FCVC <- as.numeric(obesity_data$FCVC)
obesity_data$NCP <- as.numeric(obesity_data$NCP)
obesity_data$CH2O <- as.numeric(obesity_data$CH2O)
obesity_data$FAF <- as.numeric(obesity_data$FAF)
obesity_data$TUE <- as.numeric(obesity_data$TUE)

View(obesity_data)

# Handling missing values
sum(is.na(obesity_data)) 
p <- function(x) {sum(is.na(x))/length(x)*100}   #the percentage of missing values at each column
apply(obesity_data, 2, p)
hepatitis_data[!complete.cases(hepatitis_data),]
cleaned_obesity<-na.omit(obesity_data)

# Check the importance of variables
ZeroVar <- nearZeroVar(cleaned_obesity,saveMetrics=TRUE)
nzv_sorted <- arrange(ZeroVar, desc(freqRatio))
nzv_sorted # non of variables have zero or near zero variances

# Correlation Matrix-Different correlation plots
mydata.cor <- cor(obesity_numeric) 
mydata.cor 

# 1.
corrplot_circle <- corrplot(mydata.cor,tl.col="black", tl.cex=0.8, tl.srt=70) 

# 2.
cex.before <- par("cex")
par(cex = 0.6)
corrplot(cor(mydata.cor), insig = "blank", method = "color",
         addCoef.col="grey", tl.cex = 1.2,
         cl.cex = 1.2, addCoefasPercent = TRUE,mar=c(0,0,1,0), tl.col="black",tl.srt=50)
par(cex = cex.before)

# 3.
palette <- colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = mydata.cor, col = palette, symm = TRUE)

# Drop columns 3,4,8,10
cleaned_obesity_reduced<- subset(cleaned_obesity, select = -c(8)) # dropping the 8 didnt change the accuracy

set.seed(2021) 
# Split dataset
inTraining <- createDataPartition(cleaned_obesity_reduced$class, ## indicate the outcome - helps in balancing the partitions
                                  p = .8, ## proportion used in training+ testing subset
                                  list = FALSE)
training <- cleaned_obesity_reduced[ inTraining,]
holdout  <- cleaned_obesity_reduced[-inTraining,]

# Scaling the training set (only the numeric variables, dummy variables dont need standardization)
preProcValues <- preProcess(training[c(2,3,4)], method = c("center", "scale"))
trainTransformed <- predict(preProcValues, training)

# Holdout set preparation
# Scaling the holdout set (only the numeric variables, dummy variables dont need standardization)
preProcValues <- preProcess(holdout[c(2,3,4)], method = c("center", "scale"))
holdoutTransformed <- predict(preProcValues, holdout)

#############################################################################################
# 1. Multinomial Regression
#############################################################################################
set.seed(2021)
# Parameter Tuning
control <- trainControl(method = "cv", number = 10)

set.seed(2021)
# Model Fitting
Regressionmodel <- train(class ~., 
                         data = training, 
                         method = "multinom",
                         trControl = control,
                         trace=FALSE)

# Accuracy on Training Set
mean(Regressionmodel$results$Accuracy)

# Make a Prediction on Holdout Set
Regressionmodel.pred <- predict(Regressionmodel, # predict using the fitted model
                                holdout,
                                type = "raw") ## produce only the predicted values

# Confusion Matrix
confusionMatrix(data = Regressionmodel.pred, reference = holdout$class, mode="prec_recall")
postResample(pred = Regressionmodel.pred, obs = holdout$class)

# Importance of Variables
varImp(Regressionmodel)

############################################################################################
# 2. Logistic- glmboost
############################################################################################
set.seed(2021)
# Parameter tuning
fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 10, ## k = 10
                           repeats = 10, ## and repeat 10-fold CV 10 times
                           ## Estimate class probabilities
                           savePredictions = TRUE,
                           classProbs = TRUE)

set.seed(2021)
# Model Fitting
glmBoostModel <- train(class ~ ., data=trainTransformed, 
                       method = "glmboost", 
                       trControl = fitControl,
                       tuneLength=10,
                       family="multinomial")

# Accuracy on Trainingset
glmBoostModel$results
results<- data.frame(glmBoostModel$results$Accuracy,glmBoostModel$results$Kappa)
cleaned_results <- na.omit(results)
mean(cleaned_results$glmBoostModel.results.Accuracy)
mean(glmBoostModel$results$Accuracy)

# Make a Prediction on Holdout Set
glmBoostModel.pred <- predict(glmBoostModel, # predict using the fitted model
                              holdoutTransformed,
                              type = "raw") ## produce only the predicted values

# Confusion Matrix
confusionMatrix(data = glmBoostModel.pred, reference = holdoutTransformed$class)

# Importance of Variables
varImp(glmBoostModel)


############################################################################################
# 2. LDA and QDA 
############################################################################################
set.seed(2021)

# Parameter Tuning
fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 10, ## k = 10
                           repeats = 10, ## and repeat 10-fold CV 10 times
                           ## Estimate class probabilities
                           classProbs = TRUE)
# Model Fitting
set.seed(2021)
ldamodel <- train(class ~ ., 
                  data=training, 
                  method="lda",
                  trControl = fitControl)

# Accuracy of Training Set
mean(ldamodel$results$Accuracy)

# Make a Prediction on Holdout Set
ldamodel_pred <- predict(ldamodel, 
                         holdout,
                         type = "raw") ## produce only the predicted values

# Confusion Matrix
confusionMatrix(data = ldamodel_pred, reference = holdout$class, mode="prec_recall")
postResample(pred = ldamodel_pred, obs = holdout$class)

# Importance of Variables
varImp(ldamodel)

# QDA
fitControl <- trainControl(method = "repeatedcv", ## indicate that we want to do k-fold CV
                           number = 10, ## k = 10
                           repeats = 10, ## and repeat 10-fold CV 10 times
                           classProbs = TRUE)
set.seed(2021)
qdamodel <- train(class ~ ., data=trainTransformed, 
                  method="qda",
                  trControl = fitControl)
qdamodel

varImp(ldamodel)

## Could not run qda since the dataset is small for qda, I need to eliminate some variables


############################################################################################
# 3. Knn
############################################################################################
set.seed(2021)

# Parameter Tuning
knnGrid <-  expand.grid(k = c(1:3))
fitControl <- trainControl(method = "cv",
                           number = 10, 
                           #repeats = 10, # uncomment for repeatedcv 
                           classProbs = TRUE)

# Model Fitting
knnmodel <- train(class ~ ., 
                  data = training[,-1], 
                  method = "knn",  
                  trControl = fitControl, 
                  tuneGrid = knnGrid)

# Accuracy of Training Set
mean(knnmodel$results$Accuracy)

# Make a Prediction on Holdout Set
knnmodel_pred <- predict(knnmodel, holdout,'raw')

# Confusion Matrix
confusionMatrix(data = knnmodel_pred, reference = holdout$class, mode="prec_recall")
postResample(pred = knnmodel_pred, obs = holdout$class)

# Importance of the Variables
varImp(knnmodel)

#############################################################################################
# Random Forest
#############################################################################################
set.seed (2021)

# Parameter Tuning
ftControl <- trainControl(method = "repeatedcv",
                          number = 10, 
                          repeats = 3,
                          classProbs = TRUE)

rfgrid <- expand.grid(mtry=c(1:5))

set.seed(2021)
# Model Fitting
RFmodel <- train(class~., 
                 data=training, 
                 method='rf', 
                 tuneGrid=rfgrid, 
                 trControl=ftControl,
                 n.trees=seq(100,3000,by=200),
                 savePredictions = "final")

# Accuracy on Training Set
mean(RFmodel$results$Accuracy)


# Make a Prediction on Holdout Set
RFmodel.pred <- predict(RFmodel, # predict using the fitted model
                        holdout,
                        type = "raw") ## produce only the predicted values

# Confusion Matrix
confusionMatrix(data = RFmodel.pred, reference = holdout$class, mode="prec_recall")
postResample(pred = RFmodel.pred, obs = holdout$class)

# Importance of Variables
varImp(RFmodel)

#############################################################################################
# Boosted Tree
#############################################################################################
set.seed(2021) 

# Parameter Tuning
ftControl <- trainControl(method = "repeatedcv",
                          number = 10, 
                          repeats = 3,
                          classProbs = TRUE,
                          savePredictions = "final",
)

grid <- expand.grid(interaction.depth = seq(1:3),
                    shrinkage = seq(from = 0.01, to = 0.2, by = 0.01),
                    n.trees = seq(from = 10, to = 50, by = 10),
                    n.minobsinnode = seq(from = 5, to = 20, by = 5)
)

bstreemodel <- train(class ~ ., 
                     data = training, 
                     method = "gbm",  
                     weights = NULL,
                     metric = ifelse(is.factor(y), "Accuracy", "RMSE"),   
                     maximize = ifelse(metric == "RMSE", FALSE, TRUE),
                     trControl = ftControl, 
                     tuneGrid = NULL, 
                     tuneLength = 5)

# Accuracy of Training Set
mean(bstreemodel$results$Accuracy)

# Make a Prediction on Holdout Set
bstreemodel_pred <- predict(bstreemodel, holdout)

# Confusion Matrix
confusionMatrix(data = bstreemodel_pred, reference = holdout$class, mode="prec_recall")
postResample(pred = bstreemodel_pred, obs = holdout$class)

# Importance of Variables
varImp(bstreemodel)

############################################################################################
# 6 , 7. SVM
############################################################################################
set.seed(2021)

######################## Model #1: Linear SVM #########################################
# Parameter Tuning
ftControl <- trainControl(method = "cv",
                          number = 10, 
                          #repeats = 3,
                          classProbs = TRUE,
                          savePredictions = "final",
)
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 500,1000))

set.seed(2021)
# Model Fitting
svmlinearmodel <- train(class ~ .,
                        data = training,
                        method = "svmLinear",
                        trControl = ftControl,
                        verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                        tuneGrid = grid)

# Accuracy of Training Set
mean(svmlinearmodel$results$Accuracy)

# Make a Prediction on Holdout Set
svmlinearmodel_pred <- predict(svmlinearmodel, holdout)

# Confusion Matrix
confusionMatrix(data = svmlinearmodel_pred, reference = holdout$class, mode="prec_recall")
postResample(pred = svmlinearmodel_pred, obs = holdout$class)

# Importance of Variables
varImp(svmlinearmodel)

######################## Model #2: Polynomial SVM ######################################
set.seed(2021)

# Parameter Tuning
ftControl <- trainControl(method = "cv",
                          number = 10, 
                          #repeats = 3,
                          classProbs = TRUE,
                          savePredictions = "final")

gridsvm <- expand.grid(C = c(0.01, 0.1, 10, 100,200,500,1000),
                       degree=c(1,2,3),scale=c(0.1,0.2,0.3))

# Model Fitting
svmPolymodel <- train(class ~ .,
                        data = training,
                        method = "svmPoly",
                        trControl = ftControl,
                        verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                        tuneGrid = gridsvm)

# Accuracy of Training Set
mean(svmPolymodel$results$Accuracy)


# Make a Prediction on Holdoutset
svmPolymodel_pred <- predict(svmPolymodel, holdout)

# Confusion Matrix
confusionMatrix(data = svmPolymodel_pred, reference = holdout$class, mode="prec_recall")
postResample(pred = svmPolymodel_pred, obs = holdout$class)


# Importance of Variables
varImp(svmPolymodel)

############################################################################################
# Plots
############################################################################################
# 1. Multinomial Logistic
trellis.par.set(caretTheme())
plot(Regressionmodel)

# 2. LDA
trellis.par.set(caretTheme())
plot(ldamodel)

# 3. KNN
trellis.par.set(caretTheme())
plot(knnmodel)

# 4. Random Forest 
trellis.par.set(caretTheme())
plot(RFmodel)

# 5. Boosted Tree
trellis.par.set(caretTheme())
plot(bstreemodel)

# 6. SVM-Linear
trellis.par.set(caretTheme())
plot(svmlinearmodel)

# 7. SVM-Radial
trellis.par.set(caretTheme())
plot(svmPolymodel)

############################################################################################
# Model Comparison
############################################################################################
# Model Fitting
set.seed(2021)
control <- trainControl(method = "cv", number = 10)
MltiReg <- train(class ~., 
                         data = training, 
                         method = "multinom",
                         trControl = control,
                         trace=FALSE,
                          metric="Kappa")
MltiReg$resample

set.seed(2021)
ftControl <- trainControl(method = "cv",
                          number = 10, 
                          #repeats = 3,
                          classProbs = TRUE)

rfgrid <- expand.grid(mtry=c(1:5))
RFmodel <- train(class~., 
                 data=training, 
                 method='rf', 
                 tuneGrid=rfgrid, 
                 trControl=ftControl,
                 n.trees=seq(100,3000,by=200),
                 savePredictions = "final",
                 metric="Kappa")
names(RFmodel)
RFmodel$resample


set.seed(2021)
ftControl <- trainControl(method = "cv",
                          number = 10, 
                          #repeats = 3,
                          classProbs = TRUE,
                          savePredictions = "final",
)
grid <- expand.grid(interaction.depth = seq(1:3),
                    shrinkage = seq(from = 0.01, to = 0.2, by = 0.01),
                    n.trees = seq(from = 10, to = 50, by = 10),
                    n.minobsinnode = seq(from = 5, to = 20, by = 5)
)

bstreemodel <- train(class ~ ., 
                     data = training, 
                     method = "gbm",  
                     trControl = ftControl, 
                     tuneGrid = grid, 
                     tuneLength = 5,
                     metric="Kappa")
bstreemodel$resample

resamps <- resamples(list(RF = RFmodel,
                          MltiReg=MltiReg,
                          BSTree = bstreemodel),replace = FALSE)
resamps

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))


difValues <- diff(resamps)
difValues

trellis.par.set(theme1)
bwplot(difValues, layout = c(3, 1))

# Create a matrix
conf.df <- matrix(c(0.9462,0.9531,0.9439,0.9451,0.9675,
                    0.9535,0.9539, 0.9512,0.9521,0.9717,
                    0.9828,0.9833,0.9821,0.9825,0.9897), ncol=5, byrow=TRUE)
colnames(conf.df) <- c('Accuracy','Precision','Recall', 'FScore','Balanced')
rownames(conf.df) <- c('RF','MReg','Bstrees')
conf.df <- as.table(conf.df)
conf.df

# Grouped Bar Plot
obesitybarplot <- barplot(conf.df,col = c("#00CC99", "#FFCC99","#FF9999"), 
                            beside = TRUE,
                            legend.text = c("RF", "MReg", "Bstrees"),
                            args.legend=list(cex=0.7,x="right"),
                            main="Accuracy over All Models",
                            xlab="Metrics",
                            ylab="Percentage", ylim=c(0,1.1))
percentages <- c(0.9462,0.9535,0.9828,
                 0.9531,0.9539,0.9833,
                 0.9439,0.9512,0.9821,
                 0.9451,0.9521,0.9825,
                 0.9675,0.9717,0.9897)
text(x = obesitybarplot, y = percentages, 
     label = paste(percentages*100,"%"), pos = 3.5, cex = 0.6, col = "black",srt=45)
