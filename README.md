# BIOSTAT626-Midterm01


## Task.1 - Binary Classification



### Prerequisites

Requirements for the R packages and Java envirnoment
- [Java](https://www.oracle.com/java/technologies/downloads/)
- [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html)

### Packages Installing

    install.packages('h2o')

### Model training

**(1) Baseline model**

There are three baseline models in this task. They are randomforest, gradient boosting tree,feedforward neural network, logistic regression, and naiveBayes. 
The final is a stacked ensemble learning using randomforest, grandient boosting tree, and feedforward neural network as base models with neural network as
metalearner. 

**1.Randomforest**

    Binary_rf <- h2o.randomForest(x = x,
                      y = y,
                      training_frame = training_binary,
                      ntrees = 50,
                      nfolds = nfolds,
                      keep_cross_validation_predictions = TRUE,
                      seed = 1)
**2.Gradient boosting tree**

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
**3.Feedforward neural network**

    Binary_dl = h2o.deeplearning(x = x, 
                             y = y, 
                             training_frame = training_binary,
                             hidden = c(50, 100, 50), 
                             epochs = 50,
                             nfolds = nfolds,
                             seed = 1,
                             keep_cross_validation_predictions = TRUE
                             )
**4.Logistic regression**

    GLM<-h2o.glm(x = x,
                 y = y,
                 training_frame = training_binary,
                 nfolds = 10
    )
**5.NaiveBayes**

    NB<-h2o.naiveBayes(x = x,
                       y = y,
                       training_frame = training_binary,
                       nfolds = 10
    )
Each baselines model runs on the 10-fold cross-validation. The average results of each model's performance in validation dataset are listed below:
Since I have fixed the random seed of each baselines and ensemble models, the results are reproducable. 

![Image text](https://github.com/Zongrui-Dai/BIOSTAT626-Midterm01/blob/main/result.jpg)

**(2) Final model - Stacked Ensemble model**

Taking the randomforest, gradient boosting, and Feedforward neural network trained above, a deeplearning (also feedforward neural network with default parameters) is trained as metalearner to conclude their result. The structure is listed below:

        ensemble <- h2o.stackedEnsemble(x = x, 
                                y = y, 
                                training_frame = training_binary,
                                base_models = list(Binary_dl, Binary_gbm, Binary_rf),
                                metalearner_algorithm = 'deeplearning'
                                )

To better fit the model, here I do not use 10-fold cross-validation to train the ensemble model. The performance of this model is listed below:

![Image text](https://github.com/Zongrui-Dai/BIOSTAT626-Midterm01/blob/main/stack.png)

**Performance on the Leaderboard**: This stakced ensemble model achieves 100% accuracy on the testing dataset. The upload document is named as: binary_Ayakawhen.txt



## Task.2 - Multiclass Classification

**1. Sequential Property in the activity**

By ploting the outcome of training dataset, we can see a clear time-based pattern. Static postural transition(7) always work like a transition point between two different activity types. Although we should wonder whether this pattern exists in the testing dataset (since no one can make sure the testing dataset is shuffled or not), based on the final model, this time-based pattern appears again on the testing dataset. 

![Image text](https://github.com/Zongrui-Dai/BIOSTAT626-Midterm01/blob/main/trainingy.jpg)

**2. Baseline models**

For this task, I did several baseline model to classify the problem. There are three main types: **Ensemble based model, LSTM, Conv1d+LSTM**. 
*Tips:* Training of some baseline models will be quite time consuming (especially h2o.automl). So, all these models are uploaded on the github. Using h2o.loadModel can reload them to R. 

        ## R: Load h2o model and review it's performance
        h2o.init()
        saved_model <- h2o.loadModel('E:/Biostatistics Master/BIOSTAT626/Midterm1/Multi_gbm')
        saved_model@model$cross_validation_metrics_summary

**(1) Ensemble learning**

**Why stakced ensemble learning is not choose as final model:** Based on the model above, we could easily find that some baselearner is overfitting. Since I didn't do any feature selection here, this result is expected. Choosing baselearners (GBM, RF, DL, DL+RF) as final model may have poor generalization ability since the performance on training dataset is deceptive. Also, the overfitting problem on baselearner will influence the stakced ensemble learning. If one baselearner is overfitting, metaleaner will ignore other models and put too much weight on that model.

**(2) LSTM/BILSTM**
By the Sequential Property in the training dataset, I decide to apply LSTM/BILSTM. 

**(3) Conv1D+LSTM**
Hard to become convergent. 

**3. Final model - 2Conv1D_LSTM**
![Image text](https://github.com/Zongrui-Dai/BIOSTAT626-Midterm01/blob/main/Conv1d_LSTM.jpg)


