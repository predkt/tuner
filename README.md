# # XGBoost Tuners

[XGBoost](http://xgboost.readthedocs.io/en/latest/) is an implementation of gradient boosted decision trees, popular for its scalability and superior performance. Favorite amongst Kaggle competition winners, XGBoost is equally utilized in production environments for solving a variety of classification, regression and ranking (recommendation) problems. 
Its flexibility, in part, is due to availability of large number of parameters and hyper parameters that can be tuned to control model complexity and regularization. However, tuning a large number of parameters  at  the same time is quite a challenging venture. Adding new parameters to grid search increases model builds exponentially. There are numerous blogs that share good practices for XGBoost trees such as Aarshay Jain’s guide [here](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/).

**gridsearchtuner**  builds on the Jain's techniques. Specifically, the gridsearches have been modified to perform a coarse-to-fine search: Once a best set of parameters is obtained from a cross validated grid search, a subsequent grid search is automatically performed by choosing two new values for each parameters that are closet to the chosen value within a pre-defined interval step. The limits and steps for parameters can be passed to the tuner routine. It keeps ‘fine’ tuning for a predefined number of rounds (also an input to the tuner routine), or until the parameter domain has been exhaustively searched using the predefined number of interval steps.

A sample run on the  nonMNIST data set is documented  [here](https://github.com/predkt/tuner/blob/master/gridsearchtuner/notMNIST%20with%20XGBoost.ipynb). The auto tuner creates extra 11 models to further fine tune. However, the results are far from praiseworthy. Although some gain is guaranteed, often, the gains are not significant at all. 

Regardless of the learning algorithm implemented or the technique used, what matters most, is  model performance on unseen examples, i.e, performance on the  test set, and to some extent, the validation set. Scoring against validation and test sets is a reliable technique to compare and benchmark a variety of algorithms.The run required close to 16 hours on AWS EC2 c4.8xLarge ( 36vCPUs, 60G RAM) instance and returned the following results for the best run.

![gridsearchtuner results](https://github.com/predkt/tuner/blob/master/images/gridsearchresults.png)"")

**ngTuner** is based on Andrew Ng’s recipe for tuning deep neural networks. Ng’s recipe calls for first tuning the model until training error is minimized to within a threshold. Although this usually results in an over fitted model, it ensures that we start with a model that has a high capacity, which is capable of learning enough complexity of the problem domain to explain most of the variance in the training set. In fact, it will have learned too much by overfitting. But, that is a good problem to have. Over fitting can be corrected by regularizing the model.


The tuner implements a diagnosis routine that takes in a read of training, validation and test errors and produces a diagnosis of either 'tuned', 'High Bias' or 'High Variance' verdict. Based on this diagnosis, the parameters are updated in the direction that would correct for the diagnosed problem. For instance, if the diagnosis is 'High Variance', regularization parameters such as reg_alpha and reg_lambda are moved in the positive direction (increased). The new parameters are, however, chosen randomly - by setting the existing value of the parameter along with the maximum/minimum allowed range for the parameter as the range for the new parameter.

The model diagnosis and parameters update  is done in-between training iterations by utilizing api call back functions. This means that the model can be tuned really fast, as it trains.  It also cuts down the need to create multiple models like the grid search technique. Results on the same nonMNIST problem tuned using ngTuner is documented  [here](https://github.com/predkt/tuner/blob/master/sandbox/notMNIST%20-XBGOOST%20with%20ngTuner.ipynb), This  tuner took only 17.8 seconds and yet produced a far well trained model than the 16 hour gridsearch tuner run.

![gridsearchtuner results](https://github.com/predkt/tuner/blob/master/images/ngtuner.png)"")


The tuners make use of various utility routines that provide framework like support such as versioning and saving each run, serializing and storing intermediate models and results for reruns, timer and memory diagnostic routines to monitor RAM and CPU usages, routines for producing run summaries and logs, visualizations and email reporting services. A sample summary report can be found  [here](https://github.com/predkt/tuner/blob/2325e1e4fe0fa403a8a168504c803cb0ab7880a8/sandbox/runprofiles/20171021/summary_20171021011445.txt).

This are, the widest strokes at initial proof of concept. Code cleanup and documentation will be prioritized.



