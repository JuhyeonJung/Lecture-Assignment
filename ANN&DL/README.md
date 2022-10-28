# ANN & DL

### HW1

#### 문제
- Multi-class classification problem
-	dataset : FashionMNIST
    - torchvision.datasets.FashionMNIST 활용


#### 내용
- 활용 모델  
    - logistic regression (sklearn이 아닌 pytorch로)
    - Feedfoward Neural Network (FNN의 경우, 서로 다른 3가지 hyperparameter setting 수행)
- FNN의 경우, regularization 방법 중 2가지 선택하여 반드시 포함시킬 것  
    - 성능 지표 : accuracy
    - 4가지 모델 (logistic regression 1개, FNN 3개) 중 test set 기준 가장 우수한 모델 선정

### HW2

#### 문제
-	Human Activity Recognition using smartphones data set (Multivariate timeseries classification)
-	dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones 
-	Please refer to the problem description section in the following links: 
    - https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/ 
    - https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/ 


#### 내용
- 2개 모델 활용 (세부 architecture 및 setting은 자유)
    1. LSTM layer를 활용한 classification model
    2. 1D CNN layer를 활용한 classification model

- 성능 비교
