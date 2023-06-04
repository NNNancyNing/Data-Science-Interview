# 1. General Concepts
## Bias & Variance
- **Bias**: the difference between the average prediction result of the model & the actual value.
- **Variance**: the variability of model prediction.
- **Occurance Situation**: Bias occurs when the model is too simple and can not capture the complexity of the data. A model with high bias will tend to underfit the data, and have poor performance in both the training & testing data; Variance occurs when the model is too complex and can even capture the noise of the data. A model with high variance will tend to overfit the data, and have high performance on training data, poor performance on testing data.
- **Bias & Variance tradeoff**: 
            <img src="https://github.com/NNNancyNing/Data-Science-Interview/blob/main/Images/Bias-Variance-Tradeoff.jpeg" width=550 height=400>
            
## Overfitting & Underfitting
- **Overfitting**: 

    Occurs when the model is too complex and captures not only the underlying pattern of data but also noise and randomness of data; 
    
    It can happen when training data size is too small, model is too complex, training time too long etc; 
    
    Ways to prevent overfitting: Regularization techniques(add penalty term to the loss function that encourages the model to have smaller weights: L1(Lasso Regerssion), L2(Ridge Regression)) 
    
    L1: RSS + $\lambda\Sigma_{j=1}^{p}|\beta_j|$            
    
    L2: RSS + $\lambda\Sigma_{j=1}^{p}\beta_j^2$
    
- **Underfitting**:
   
    Occurs when the model is too simple and didn't successfully capture important features. Will perform poor on both training & testing data. 
    
    Ways to prevent underfitting: increase training data size, increase model complexity.


## Regression Metrics
- **MSE**: average squared difference between predicted and actual values. MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$
- **RMSE**: square root of MSE. RMSE = $\sqrt{\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2}$
- **MAE**: average absolute difference between predicted and actual values. MAE = $\frac{1}{n} \Sigma_{i=1}^n|{y}-\hat{y}|$
- **R-Square**: the proportioin of variance in the target variable that is explained by the model. R-Square = $1-\frac{\Sigma_{i=1}^{n}(y_i - \hat{y_i})^2}{\Sigma_{i=1}^{n}(y_i - \bar{y})^2}$
- **Adjusted R-Square**: similar to R-Square but penalize models with too many predictors. Adj R-Squre = $1-\frac{(1-R^2)(n-1)}{n-p-1}$

- RMSE is preferred over MSE, as it's the same unit as the target variable; MAE is less sensitive to outliers; Adjusted R-Square is useful when comparing models with different number of predictors.

## Classification Metrics
- **Accuracy**
- **Precision**: the proportion of true positive classifications out of all positive classifications. TP/(TP+FP); Used when cost of FP is high.
- **Recall**: the proportion of true positive classifications out of all actual positive instances. TP/(TP+FN); Used when cost of FN is high.
- **F1 Score**: harmonic mean of Precision and Recall. $2\frac{precisionxrecall}{precision+recall}$; Used when cost of FP and FN is similar.
- **ROC Curve**: the plot of TPR and FPR; Used when cost of FP and FN varies.
- **AUC**: area under ROC Curve
- **Log Loss(Cross-entropy Loss)**: difference between the predicted probabilities and actual binary labels; Used when model outputs probabilities instead of binary classifications.
- **Confusion Matrix**: 
            <img src="https://github.com/NNNancyNing/Data-Science-Interview/blob/main/Images/confusion%20matrix.jpeg" width=550 height=400>


## General Machine Learning Steps
- **Whole Steps**: 
            
    Split train&test dataset; 
    
    Feature engineering; 
    
    Specify a resampling procedure: K-fold CV/bootstrap; 
    
    Create the Hyperparameter Grid; 
    
    Execute grid search; 
    
    Evaluate performance to choose the best hyperparameter combinations; 
    
    Apply model on test dataset to obtain classification/regression results.

- **K-fold Cross Validation**: Randomly divide training set into K folds with equal size. Fit the model on K-1 folds, assess the performance on the remaining fold. The above procedure repeats k times, when we obtain k validation errors, we take average of these k errors. 

- **Bootstrap Sampling**: A boostrap sample is a random sample of the data taken with replacement. Thus it has the same size, also the same distribution of values as the original data set; the model is validated on the OOB samples(observations not contained in a bootstrap sample)


## General Feature Engineering Steps
- **FE Steps**:

    Feature filtering;
    
    Dealing with missingness(Imputation);
    
    Numerical feature engineering(Normalization(deal with skewness/outliers), Standardization(deal with wide range));
    
    Categorical feature engineering(Lumping, One-hot encoding, label encoding)
    
- **Normalization & Standardization**:
    
    Normalization: scale data so that it falls between 0 and 1.  $X_i` = (X_i - min(X))/(max(X)-min(X))$
    
    Standardization: transform data so that it has mean 0 and standard deviation 1 $X_i` = (X_i - mean(X)) / std(X)$
    
    
- **One-hot encoding & Dummy encoding**:
    
    One-hot encoding created a new set of binary variables that is equal to the number of categories (k) in the variable.
    
    Dummy encoding created k-1 binary variables.
    
     <img src="https://github.com/NNNancyNing/Data-Science-Interview/blob/main/Images/One-hot-encoding.png" width=450 height=300>
     <img src="https://github.com/NNNancyNing/Data-Science-Interview/blob/main/Images/Dummy-encoding.png" width=450 height=300>
     
     
    
   
 

