
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
y1 = df["class"].values

#********************************************


#Split data into train and test to verify accuracy after fitting the model. 

X = df.drop(labels = ["class", "subclass"], axis=1) 

#DEALING WITH IMBALANCE USING SMOTE
smote = SMOTE()

x_smote, y_smote = smote.fit_resample(X, y1)

print('Original dataset shape', Counter(y1))
print('Resample dataset shape', Counter( y_smote))


# Encode labels into categorical format
label_encoder = LabelEncoder()
y= label_encoder.fit_transform(y1)

#*******************************************
# Number of folds for cross-validation
num_folds = 5

# Initialize the KFold cross-validation object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
accuracy_per_fold = []
accuracy_per_epoch_per_fold = []
confusion_matrices = []
roc_auc_per_fold = []

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{num_folds}")
    
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    

    model = Sequential()
   
    model.add(Convolution1D(64, kernel_size=122, padding="same", activation="relu", input_shape=(122, 1)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape=(128, )))

    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))

    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    
    accuracy_per_epoch = []
    
    # Train the model and track accuracy per epoch
    for epoch in range(10):  # Modify the number of epochs as needed
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracy_per_epoch.append(accuracy)
        print(f"Fold {fold + 1}/{num_folds} - Epoch {epoch + 1}/{10} - Accuracy: {accuracy:.4f}")

    accuracy_per_fold.append(accuracy)
    accuracy_per_epoch_per_fold.append(accuracy_per_epoch)
    
    # Predict probabilities for ROC curve
    y_pred_prob = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    # Compute ROC curve and ROC AUC
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
    roc_auc = auc(fpr, tpr)
    roc_auc_per_fold.append(roc_auc)
    
    # Predict classes for confusion matrix
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(accuracy_per_fold)
print("Mean Accuracy:", mean_accuracy)

# Plot accuracy vs. epoch for each fold
plt.figure(figsize=(10, 6))
for fold, accuracies in enumerate(accuracy_per_epoch_per_fold):
    plt.plot(range(1, 11), accuracies, label=f"Fold {fold + 1}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch per Fold")
plt.legend()
plt.show()

# Display accuracy per fold
for fold, accuracy in enumerate(accuracy_per_fold):
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

# Calculate and plot overall confusion matrix
overall_cm = np.sum(confusion_matrices, axis=0)
plt.figure(figsize=(8, 6))
plt.imshow(overall_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Overall Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(5), np.arange(5))
plt.yticks(np.arange(5), np.arange(5))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Calculate and plot ROC curve and ROC AUC
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip([mean_fpr] * num_folds, roc_auc_per_fold)], axis=0)
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.2f})'.format(np.mean(roc_auc_per_fold)))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()