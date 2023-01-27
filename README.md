# Diabetes prediction analysis

Analyze [Kaggle dataset](https://www.kaggle.com/datasets/prosperchuks/health-dataset), using ML and DL approaches, to find out the relations between features and targets.

## Data Pre-processing and Visualization
(1) This dataset has no null values and all the data type are already in the form of int64. The dataset has 70692 data and includes 15 features.   

(2) Visualize the corelation between each features and targets.
<img width="708" alt="截圖 2023-01-27 下午9 11 34" src="https://user-images.githubusercontent.com/68526411/215095097-7725bfcd-dd01-4a32-849b-10a610d164e2.png"> 
 
It's a pity that most of the data are binary and that there are no really high correlation features in the dataset, GenHlth may be the one which have higher correlation. So I also look at the correlation graph between the features and the target to pick out the three most high correlation feature(Age, HighChol, GenHlth) to do further prediction.

(3) Scale the data(Standard scale and Min Max scale) to fit in the model so that every feature are in the similar scale. Later I only use Standard scale since most of the data are already binary(0 and 1).

## Algorithms and framework
PyTorch models(Logistic Regression and Sequential Neural Network), Scikit-learn models(LogisticRegression, SVM(linear), SVM(rbf), KNN, Decision Tree Classifier, Perceptron, Random Forest Classifier) to predicting the baseline, after scaling, and after feature selections to analyze.

## Results
The performance of models turns out not to improve much after doing some feature selection, scaling and tuning the hyper parameters(neighbor of knn, n_estimator, learning rate, epoch of deep learning model, etc.). The accuracy are mostly 0.7. This result may due to the binary value of the features(11 out of 15 are binary). Even so, there are still some interesting results I will like to share.

#### Feature Selection: 
After doing feature selection picking out 3 highest correlation feature with coef 0.3 and above using scikit-learn, I found that Decision tree and perceptron both have improve much(accuracy 0.5 to 0.7), whereas others do not. I have tried to pick out less feature but picking out 3 performs the best. This could be possible because the more features there are, the more complex the model will be for these two simple models. Take decision tree as example, the three will have a very deep depth if there are a lot of features, this will lead to a poor performance of the model. 

#### Scaling:
After doing standard scaling to scale all of the features into Gaussian distribution with 0 mean and unit variance. I found it very interesting because Logistic Regressio has a substantial improve(accuracy 0.5 to 0.78), even from the first epoch. This may probably due to the calculation of the logistic regression. Logistic regression or even SVM model uses the feature value and correlation to calculate the dot product to predict the final result. Therefore, feature with large value may dominate the prediction and this is really a flaw of these models which calculate the dot product. Scaling the feature may help mitigate this problem since all features are on the similar scale and all the feature may contribute equally to the performance of the model. 
<img width="541" alt="截圖 2023-01-03 上午10 53 49" src="https://user-images.githubusercontent.com/68526411/215100815-877a2fcc-0ccc-44d4-a9e3-0d83b226e646.png">
