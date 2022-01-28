
# NYC Tree Health Predictor

This project creates a multi-class classifier from a
tree censeus in NYC, specifically the Bronx, which contains
over 65,000 data of individual trees. This model
predicts health conditions of trees in NYC to improve 
environmental regulations.


## Acknowledgements

 - [Aurélien Géron - Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow](https://github.com/ageron/handson-ml2)
 - [Pipelines & Custom Transformers in scikit-learn: The step-by-step guide (with Python code)](https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156)
 - [Gradient Boosting Classifiers in Python with Scikit-Learn](https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/)
 - [Support Vector Machine - Introduction to Machine Learning Algorithms](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)

## Authors

- [@eugenekim3107](https://www.github.com/eugenekim3107)


## Roadmap

- Explorartory Data Analysis: Extract necessary features from data set, Check for unusual patterns between features, Examine missing or duplicate data points

- Data Preprocessing: Stratify data set using relevant feature, Replace missing data points respectively with custom transformers, Create separate pipelines for outliers or data type conversions

- Model Training and Testing: Select multi-class classifier models to train with data set, Tune hyperparameters for better performance or less chance of overfitting

- Evalutation: Choose scoring method and determine model with the highest score, Create visuals of scores using heat maps and bar charts, Reanalyze project for improvements


## Optimizations

Hyperparameter tuning: K-Nearest Neighbor classifier's "metric" and "n_neighbors" were changed from the default parameters to "manhattan" for "metric" and "10" for "n_neighbors"

## Test Performance

![KNN Classifier](https://i.postimg.cc/VLpzsyvb/Screen-Shot-2022-01-27-at-4-12-49-PM.png)

![KNN Classifier](https://i.postimg.cc/V6f8ZmG2/Screen-Shot-2022-01-27-at-4-12-55-PM.png)