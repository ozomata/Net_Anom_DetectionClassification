
import sys
import numpy as np 
from numpy import where
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from scikeras.wrappers import KerasClassifier
from itertools import cycle

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, Dense, Dropout, Activation
from keras.utils import to_categorical
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV
print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)

#**************************
#Reading Dataset
df1 = pd.read_csv('UNSW_NB15_testing-set.csv')
df2 = pd.read_csv('UNSW_NB15_training-set.csv')
df1 = df1.drop('id', axis=1)
df1 = df1.drop('label', axis= 1)
df2 = df2.drop('id', axis= 1)
df2 = df2.drop('label', axis= 1)

df = pd.concat([df1,df2])

# temporarily remove attack class column
tmp = df.pop('attack_cat')


# Create a one-hot encoder
def one_hot(df):
    columns_to_encode = ['proto','state','service'] 
    for each in columns_to_encode:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each,axis =1)
    return df

df = one_hot(df)
#****************************

df['class']=tmp

X = df.drop(labels = ["class"], axis=1) 
#Define the dependent variable that needs to be predicted (labels)
y = df["class"].values

#DEALING WITH IMBALANCE USING SMOTE
smote = SMOTE()

x_smote, y_smote = smote.fit_resample(X, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter( y_smote))


# Encode labels into categorical format
label_encoder = LabelEncoder()
Y= label_encoder.fit_transform(y)




#************************************


# Defining Model
# Defining Model
def CNNbilstm(optimizer='adam', dropout_rate=0.5):
   
    model = Sequential()
    model.add(Convolution1D(filters=64, kernel_size=64, padding="same",activation="relu", input_shape=((196,1))))
    model.add(MaxPooling1D(pool_size=(10)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape = (128, )))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# Create a KerasClassifier for scikit-learn
model = KerasClassifier(model=CNNbilstm, verbose=1)

# Define the hyperparameter grid
param_grid = {
    #'model__optimizer': ['adam', 'sgd'],  # You can add more optimizers to test
    #'neurons':[128, 64],
    'model__dropout_rate': [0.2, 0.5],  # You can add more dropout rates to test
    'epochs': [10, 20],  # You can add more epochs to test
    'batch_size': [32, 64]  # You can add more batch sizes to test
}

#two hyperparameters: neurons (number of neurons in the hidden layer) and dropout_rate (dropout rate for the dropout layer).
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3 )
grid_result = grid.fit(X, Y)

# summarize results
print("Best accuracy of: %f using %s" % (grid_result.best_score_, 
                                         grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

###########################################################

#Let us load the best model and predict on our input data
best_model =grid_result.best_estimator_

# Predicting the Test set results
y_pred = best_model.predict(X)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, y_pred)
sns.heatmap(cm, annot=True)

