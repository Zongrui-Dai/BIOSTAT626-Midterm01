# BIOSTAT626-Midterm01


## Task.1 - Binary Classification



### Prerequisites

Requirements for the R packages and Java envirnoment
- [Java](https://www.oracle.com/java/technologies/downloads/)
- [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html)

### Packages Installing

    install.packages('h2o')

### Model training

(1) Baseline model
There are three baseline models in this task. They are randomforest, gradient boosting tree,feedforward neural network, logistic regression, and naiveBayes. 
The final is a stacked ensemble learning using randomforest, grandient boosting tree, and feedforward neural network as base models with neural network as
metalearner. 

*1.Randomforest*

    Binary_rf <- h2o.randomForest(x = x,
                      y = y,
                      training_frame = training_binary,
                      ntrees = 50,
                      nfolds = nfolds,
                      keep_cross_validation_predictions = TRUE,
                      seed = 1)
*2.Gradient boosting tree*

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
*3.Feedforward neural network*

    Binary_dl = h2o.deeplearning(x = x, 
                             y = y, 
                             training_frame = training_binary,
                             hidden = c(50, 100, 50), 
                             epochs = 50,
                             nfolds = nfolds,
                             seed = 1,
                             keep_cross_validation_predictions = TRUE
                             )
*4.Logistic regression*

    GLM<-h2o.glm(x = x,
                 y = y,
                 training_frame = training_binary,
                 nfolds = 10
    )
*5.NaiveBayes*

    NB<-h2o.naiveBayes(x = x,
                       y = y,
                       training_frame = training_binary,
                       nfolds = 10
    )
Each baselines model runs on the 10-fold cross-validation. The average results of each model's performance in validation dataset are listed below:

![Image text](https://github.com/Zongrui-Dai/BIOSTAT626-Midterm01/result.jpg)




## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

Explain what these tests test and why

    Give an example

### Style test

Checks if the best practices and the right coding style has been used.

    Give an example

## Deployment

Add additional notes to deploy this on a live system

## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
