## This R is used to represent the training process of each baseline model in h2o
## It can be wasted of time. You can just load the trained model directly through the 
## 'Fast_verification_Multiclass_h2o.Rmd'
library(h2o)
library(ggplot2)

Sys.setenv(JAVA_HOME="D:/Java")
setwd('E:/Biostatistics Master/BIOSTAT626/Midterm1/')
training<-read.table('training_data.txt',head=T)
testing<-read.table('test_data.txt',head=T)

h2o.init()
saved_model <- h2o.loadModel('E:/Biostatistics Master/BIOSTAT626/Midterm1/Multi_gbm')
saved_model@model$cross_validation_metrics_summary

d<-read.csv('E:/Biostatistics Master/BIOSTAT626/Midterm1/tsne.csv')
d$y<-as.factor(y)
ggplot(d,aes(x=X0,y=X1,col=y,group=y))+
  geom_point()+
  ggtitle('tSNE Cluster of training dataset')

## The training_muticlass is also output for python to train LSTM
training_muticlass<-training
y<-training_muticlass$activity
y[y>=7]=7
training_muticlass$y<-as.factor(y)

y<-training_muticlass$y
ymatrix<-matrix(y,ncol = 3,byrow = T)

yd<-data.frame(y,x=1:7767)
ggplot(yd,aes(x=x,y=y))+
  geom_point(col=as.factor(y))+
  geom_line()

############################################################################
y1 <- "y"
x1 <- setdiff(names(training_muticlass), c('activity','subject','y'))
testing<-as.h2o(testing)
training_muticlass<-as.h2o(training_muticlass)

## GBM training
nfolds <- 10
Multi_gbm <- h2o.gbm(x = x1,
                     y = y1,
                     training_frame = training_muticlass,
                     ntrees = 50,
                     max_depth = 3,
                     min_rows = 2,
                     learn_rate = 0.2,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     seed = 1)
Perform_gbm <- h2o.performance(Multi_gbm, newdata = training_muticlass)
model_path <- h2o.saveModel(object = Multi_gbm, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Multi_gbm',
                            force = TRUE)
## RF training
Multi_rf <- h2o.randomForest(x = x,
                             y = y,
                             training_frame = training_muticlass,
                             ntrees = 50,
                             nfolds = nfolds,
                             keep_cross_validation_predictions = TRUE,
                             seed = 1)
Perform_rf <- h2o.performance(Multi_rf, newdata = training_muticlass)
model_path <- h2o.saveModel(object = Multi_rf, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Multi_rf',
                            force = TRUE)

## DL training
Multi_dl = h2o.deeplearning(x = x, 
                            y = y, 
                            training_frame = training_muticlass,
                            hidden = c(50, 100, 50), 
                            epochs = 50,
                            nfolds = nfolds,
                            seed = 1,
                            keep_cross_validation_predictions = TRUE
)
Perform_dl <- h2o.performance(Multi_dl, newdata = training_muticlass)
model_path <- h2o.saveModel(object = Multi_dl, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Multi_dl',
                            force = TRUE)

## Ensemble
ensemble <- h2o.stackedEnsemble(x = x, 
                                y = y, 
                                training_frame = training_muticlass,
                                base_models = list(Multi_dl, Multi_gbm, Multi_rf),
                                metalearner_algorithm = 'deeplearning'
)

gbm_binary_class<-h2o.predict(Multi_gbm,testing)
rf_binary_class<-h2o.predict(Multi_rf,testing)
dl_binary_class<-h2o.predict(Multi_dl,testing)
gbm_binary_class = as.data.frame(gbm_binary_class)
rf_binary_class = as.data.frame(rf_binary_class)
dl_binary_class = as.data.frame(dl_binary_class)

Data_ensemble<-as.data.frame(h2o.predict(ensemble,testing))
Data_baselearner<-data.frame(gbm = gbm_binary_class$predict,
                             rf = rf_binary_class$predict,
                             dl = dl_binary_class$predict,
                             ensemble = Data_ensemble$predict
)
write.csv(Data_baselearner,'Multi_baselearner.csv')

## Auto_ML
Muticlass_ml<-h2o.automl(x = x,
                         y = y,
                         training_frame = training_muticlass,
                         nfolds = 10,
                         keep_cross_validation_predictions = TRUE,
                         max_models = 50,
                         seed = 1)
m2 <- Muticlass_ml@leader
Perform_ml <- h2o.performance(m2, newdata = training_muticlass)
model_path <- h2o.saveModel(object = m2, 
                            path = 'E:/Biostatistics Master/BIOSTAT626/Midterm1/Stack_Mutiml',
                            force = TRUE)
BEST_ML<-as.data.frame(h2o.predict(m2,testing))
write.csv(BEST_ML,'E:/Biostatistics Master/BIOSTAT626/Midterm1/Muti_BEST_ML.csv')



