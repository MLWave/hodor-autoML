# hd.stacking.blend_proba

> function hd.stacking.blend_proba(clf, X_train, y, X_test, nfolds=5, save_preds="", save_test_only="",save_params="", clf_name="XX", generalizers_params=[], minimal_loss=0,return_score=False,minimizer="log_loss")

## A Cross-Validated Blender.

This blender creates a number of stratified folds from a train set. It then trains a classifier on every fold and creates predictions for a holdout set. This way predictions for the entire training are created.

Finally it creates the predictions for the test set from the entire training set. These probabilities can be used to created blended datasets which can be employed by higher-level models to reduce the generalization error.

### Parameters: 

#### Required
*clf*: A scikit-learn classifier or scikit-learn valid API wrapper.
*X_train*: NumPy Array. The training set.
*y*: NumPy Array. The target labels.
*X_test*: Numpy Array. The test set.

#### Optional
*save_preds*: Path of directory where to save the probability predictions for test and train sets. Format of saved files is numpy array.
On Windows: `"kaggle_timesuck\\generalizers\\"`
*save_test_only*: Path of directory where to save the probability predictions for test set only. Format of saved file is numpy array. Default: "" (empty string means nothing will be saved)
*save_params*: Path of directory where to save the model parameters. Format of saved file is json. Default: "" (empty string means nothing will be saved)
*clf_name*: Identifier for model, eg: `"RF"`. Is used in filename for any saved files. Default: `"XX"` (means 4 first letters of classifier object description will be used)
*return_score*: Boolean. When True returns X_train, X_test, cv_loss else returns only X_train, X_test. Default: False. 
*minimizer*: String. The metric you want to minimize. `["log_loss","accuracy"]`. Default "log_loss"
*nfolds*: Int. Number of stratified folds to create.

#### Experimental
*generalizer_params*: Glob of generalizer models used by a stacker, eg: generalizers\\*.json
*minimal_loss*: Return False if first-fold cross-validation score is higher than specified loss. Default: 0 (0 means no False will be returned)

## References:
Heikki Huttunen et al (2015) Computer Vision for Head Pose Estimation: Review of a Competition
sklearn.ensemble: Ensemble Methods http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble