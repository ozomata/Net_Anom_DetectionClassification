"""""
Ozomata Asun  
26218999

Aknowledgment: Part of the dataprocessing and the model definition were gotten from the work of (Sinha and Manollas, 2020)Efficient-CNN-BiLSTM-for-Network-IDS

"""""
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
from keras.models import save_model

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, Dense, Dropout, Activation
from keras.utils import to_categorical
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from collections import Counter
import time
print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)
seconds = time.time()
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


# Labbel Encoding
label_encoder = LabelEncoder()
Y= label_encoder.fit_transform(y)

#*******************************************

# Defining  the classifier model
def cnnbilstm():
   
    model = Sequential()
    model.add(Convolution1D(64, kernel_size=122, padding="same",activation="relu",input_shape=(122, 1)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape = (128, )))

    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))

    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# create model
my_model = KerasClassifier(model=cnnbilstm, epochs=1, batch_size=64, verbose=1)

# Create the model
my_model1 = cnnbilstm()

# Print the model summary
my_model1.summary()


#*********************************
 # Define pipeline to include scaling and the model
steps = list()
steps.append(('scaler', MinMaxScaler()))  # Adding MinMaxScaler here
steps.append(('classifier', my_model))
pipeline = Pipeline(steps=steps)


# Evaluate the model using cross-validation
cv = StratifiedKFold(n_splits=5, random_state=17, shuffle=True)

# Lists to store evaluation metrics for each fold
accuracies = []
f1_scores = []
precisions = []
conf_matrices = []
recalls =[]

for train_idx, test_idx in cv.split(X, Y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    
        # Fit the pipeline on training data
    history=pipeline.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = pipeline.predict(X_test)
    print("\n operation time: = ",time.time()- seconds ,"secs \n")


    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    
    accuracies.append(accuracy)
    f1_scores.append(f1)
    precisions.append(precision)
    conf_matrices.append(conf_matrix)
    recalls.append(recall)

# Predict probabilities for each class
y_prob = pipeline.predict_proba(X_test)

# Save the model
   
my_model1.save('model.h5')


# Compute ROC curve and AUC for each class
n_classes = len(label_encoder.classes_)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Calculate mean accuracy, F1 score, and precision across folds
mean_accuracy = np.mean(accuracies)
mean_f1 = np.mean(f1_scores)
mean_precision = np.mean(precisions)
recall = np.mean(recalls)

# Print accuracy for each fold
for fold_num, accuracy in enumerate(accuracies, start=1):
    print(f"Accuracy for Fold {fold_num}: {accuracy}")

# Print results
print("Mean Accuracy: ", mean_accuracy)
print("Mean F1 Score: ", mean_f1)
print("Mean Precision: ", mean_precision)
print("recall: ", recall)
# Calculate and print overall confusion matrix


# Generate classification report
class_names = label_encoder.classes_
classification_rep = classification_report(Y, pipeline.predict(X), target_names=class_names)
print("Classification Report:\n", classification_rep)

# Create a dataframe to store evaluation results
evaluation_results = pd.DataFrame({
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'Precision': precisions,
    'recall' : recalls

})

# Create a bar plot for accuracy per fold
sns.barplot(x=np.arange(1, len(accuracies)+1), y=accuracies)
# Set the y-axis limits to start at y=0.95
plt.ylim(0.95, max(accuracies) + 0.05)  # Adjust the upper limit as needed
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy per Fold')
plt.show()

sns.countplot(x='class', data=df)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()



sns.lineplot(data=evaluation_results)
plt.xlabel('Fold')
plt.ylabel('Metric Value')
plt.title('Evaluation Metrics Trends')
plt.xticks(rotation=45)
plt.show()

# Calculate and print overall confusion matrix
overall_conf_matrix = sum(conf_matrices)

# Calculate and print overall confusion matrix with class names
cm = sum(conf_matrices)
print("Overall Confusion Matrix:\n", cm)


cm_names = pd.DataFrame(cm, index=class_names, columns=class_names) # Create a DataFrame for the confusion matrix


# Visualize the confusion matrix with Seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm_names, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Class Names')
plt.show()

# Define class labels
class_labels = ['Dos', 'Normal', 'Prob_Pred', 'R2L', 'U2R']

# Calculating the TP
true_positives = [cm[i][i] for i in range(5)]
false_negatives = [sum(cm[i]) - true_positives[i] for i in range(5)]

# Applying formula for DR
detection_rates = [true_positives[i] / (true_positives[i] + false_negatives[i]) for i in range(5)]

# Print DR /class
for i, rate in enumerate(detection_rates):
    print(f"{class_labels[i]}: Detection Rate = {rate:.4f}")

# Calculate ing FP and TN per class
false_positives = [sum(cm[i]) - cm[i][i] for i in range(5)]
true_negatives = [sum(cm[j][i] for j in range(5)) - cm[i][i] for i in range(5)]

# Calculatinf FPR
false_positive_rates = [false_positives[i] / (false_positives[i] + true_negatives[i]) for i in range(5)]

# Print FPR /class
for i, rate in enumerate(false_positive_rates):
    print(f"{class_labels[i]}: False Positive Rate = {rate:.4f}")

average_detection_rate = sum(detection_rates) / len(detection_rates)
average_fpr = sum(false_positive_rates) / len(false_positive_rates)

# Print the Average Detection Rate and Average FPR
print(f"Average Detection Rate = {average_detection_rate:.4f}")
print(f"Average False Positive Rate = {average_fpr:.4f}")