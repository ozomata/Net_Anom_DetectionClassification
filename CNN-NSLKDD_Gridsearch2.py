
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


#*******************************************
#read datases
pd.set_option('display.max_columns',None)
df1 = pd.read_csv('KDDTrain+.txt',header=None)
df2 = pd.read_csv('KDDTest+.txt', header=None)
df = pd.concat([df1,df2])
#*******************************************
#Adding column headers  Available in the Dataset download
df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']


df = df.drop('difficulty_level', axis= 1) # removing unecessary column
tmp = df["subclass"]  #return sub class odf attacks to dataframe

df.isnull().values.any()   #Just checkin :-) This MSc has been Crazzyyyyyyy
#*******************************************

# Create a one-hot encoder
def one_hot(df):
    columns_to_encode = ['protocol_type','service','flag'] 
    for each in columns_to_encode:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each,axis =1)
    return df

df = one_hot(df)


#*******************************************
#Categorising Attacks in to 4 buckets
classlist = []
check1 = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")
check2 = ("ipsweep","mscan","nmap","portsweep","saint","satan")
check3 = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")
check4 = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")

DoSCount=0
ProbeCount=0
U2RCount=0
R2LCount=0
NormalCount=0

for item in tmp:
    if item in check1:
        classlist.append("DoS")
        DoSCount=DoSCount+1
    elif item in check2:
        classlist.append("Probe")
        ProbeCount=ProbeCount+1
    elif item in check3:
        classlist.append("U2R")
        U2RCount=U2RCount+1
    elif item in check4:
        classlist.append("R2L")
        R2LCount=R2LCount+1
    else:
        classlist.append("Normal")
        NormalCount=NormalCount+1
        
        
        
#*******************************************
df["class"] = classlist

df["class"].value_counts()  #just Checking again

# temporarily remove attack class column
y = df["class"].values

#********************************************


#Split data into train and test to verify accuracy after fitting the model. 

X = df.drop(labels = ["class", "subclass"], axis=1) 

#DEALING WITH IMBALANCE USING SMOTE
smote = SMOTE()

x_smote, y_smote = smote.fit_resample(X, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter( y_smote))


# Encode labels into categorical format
label_encoder = LabelEncoder()
Y= label_encoder.fit_transform(y)

#*******************************************

# Defining Model
def create_model(optimizer='adam', dropout_rate=0.5, neurons=128):
   
    model = Sequential()
    model.add(Convolution1D(64, kernel_size=122, padding="same",activation="relu",input_shape=(122, 1)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape = (128, )))

    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))

    model.add(Dropout(dropout_rate))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model


# Create a KerasClassifier for scikit-learn
model = KerasClassifier(model=create_model, verbose=1)



param_grid = {'neurons': [128,64],
              'batch_size': [4],
              'optimizer': ['SGD', 'RMSprop']
              } 

# 3 x 2 x 3 = 18 combinations for parameters

#Define the model using KerasClassifier method.
#This makes our keras model available for GridSearch
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, verbose=1)

#n_jobs=-1 parallelizes but it may crash your system. 
#Provide the metric for KFold crossvalidation. cv=3 is a good starting point
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)

#Takes a long time based on the number of parameters and cv splits. 
#In our case - 18 * 3 * 2 * num_epochs = 1080 total epochs if epochs=10
zzz
# summarize results
print("Best accuracy of: %f using %s" % (grid_result.best_score_, 
                                         grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))






