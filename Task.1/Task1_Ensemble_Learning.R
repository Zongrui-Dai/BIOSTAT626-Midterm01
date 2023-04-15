## Make sure the Java and H2o have been setting up
library(h2o)

Sys.setenv(JAVA_HOME="D:/Java")
setwd('E:/Biostatistics Master/BIOSTAT626/Midterm1/')
training<-read.table('training_data.txt',head=T)
testing<-read.table('test_data.txt',head=T)

h2o.init()

### Binary classification
training_binary<-training
y<-training_binary$activity
y[y<=3]=1
y[y>3]=0
training_binary$y<-as.factor(y)

y <- "y"
x <- setdiff(names(training_binary), c('activity','subject','y'))
testing<-as.h2o(testing)
training_binary<-as.h2o(training_binary)

# GBM
nfolds <- 10
Binary_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = training_binary,
                  ntrees = 50,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)
Perform_gbm <- h2o.performance(Binary_gbm, newdata = training_binary)
model_path <- h2o.saveModel(object = Binary_gbm, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Binary_gbm',
                            force = TRUE)
## RF
Binary_rf <- h2o.randomForest(x = x,
                      y = y,
                      training_frame = training_binary,
                      ntrees = 50,
                      nfolds = nfolds,
                      keep_cross_validation_predictions = TRUE,
                      seed = 1)
Perform_rf <- h2o.performance(Binary_rf, newdata = training_binary)
model_path <- h2o.saveModel(object = Binary_gbm, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Binary_rf',
                            force = TRUE)

## DL
Binary_dl = h2o.deeplearning(x = x, 
                             y = y, 
                             training_frame = training_binary,
                             hidden = c(50, 100, 50), 
                             epochs = 50,
                             nfolds = nfolds,
                             seed = 1,
                             keep_cross_validation_predictions = TRUE
                             )
Perform_dl <- h2o.performance(Binary_dl, newdata = training_binary)
model_path <- h2o.saveModel(object = Binary_dl, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Binary_dl',
                            force = TRUE)

## Ensemble
ensemble <- h2o.stackedEnsemble(x = x, 
                                y = y, 
                                training_frame = training_binary,
                                base_models = list(Binary_dl, Binary_gbm, Binary_rf),
                                metalearner_algorithm = 'deeplearning'
                                )


gbm_binary_class<-h2o.predict(Binary_gbm,testing)
rf_binary_class<-h2o.predict(Binary_rf,testing)
dl_binary_class<-h2o.predict(Binary_dl,testing)
gbm_binary_class = as.data.frame(gbm_binary_class)
rf_binary_class = as.data.frame(rf_binary_class)
dl_binary_class = as.data.frame(dl_binary_class)

Data_ensemble<-as.data.frame(h2o.predict(ensemble,testing))
Data_baselearner<-data.frame(gbm = gbm_binary_class$predict,
                             rf = rf_binary_class$predict,
                             dl = dl_binary_class$predict,
                             ensemble = Data_ensemble$predict
                             )
write.csv(Data_baselearner,'Binary_baselearner_final.csv')

## Other baseline models that are not include in the ensemble model
GLM<-h2o.glm(x = x,
        y = y,
        training_frame = training_binary,
        nfolds = 10
)

NB<-h2o.naiveBayes(x = x,
             y = y,
             training_frame = training_binary,
             nfolds = 10
)

GBM_CV<-data.frame(Binary_gbm@model$cross_validation_metrics_summary)[c(1,2,6,9),1]
RF_CV<-data.frame(Binary_rf@model$cross_validation_metrics_summary)[c(1,2,6,9),1]
DL_CV<-data.frame(Binary_dl@model$cross_validation_metrics_summary)[c(1,2,6,9),1]
GLM_CV<-data.frame(GLM@model$cross_validation_metrics_summary)[c(1,2,6,9),1]
NB_CV<-data.frame(NB@model$cross_validation_metrics_summary)[c(1,2,6,9),1]

Results<-data.frame(GBM_CV,RF_CV,DL_CV,GLM_CV,NB_CV)
rownames(Results)<-c('Accuracy','AUC','F1','Logloss')
print(Results)





