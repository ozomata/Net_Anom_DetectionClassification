import sys
import numpy as np 
from numpy import where
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, confusion_matrix, roc_curve, auc, roc_auc_score,classification_report
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
def cnnbilstm():
   
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

#


# create model
my_model = KerasClassifier(model=cnnbilstm, epochs=10, batch_size=32, verbose=1)

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

for train_idx, test_idx in cv.split(X, Y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    
        # Fit the pipeline on training data
    history=pipeline.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    accuracies.append(accuracy)
    f1_scores.append(f1)
    precisions.append(precision)
    conf_matrices.append(conf_matrix)

# Predict probabilities for each class
y_prob = pipeline.predict_proba(X_test)




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

# Print accuracy for each fold
for fold_num, accuracy in enumerate(accuracies, start=1):
    print(f"Accuracy for Fold {fold_num}: {accuracy}")

# Print results
print("Mean Accuracy:", mean_accuracy)
print("Mean F1 Score:", mean_f1)
print("Mean Precision:", mean_precision)
 
# Calculate and print overall confusion matrix
overall_conf_matrix = sum(conf_matrices)
print("Overall Confusion Matrix:\n", overall_conf_matrix)

# Generate classification report
class_names = label_encoder.classes_
classification_rep = classification_report(Y, pipeline.predict(X), target_names=class_names)
print("Classification Report:\n", classification_rep)

# Create a dataframe to store evaluation results
evaluation_results = pd.DataFrame({
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'Precision': precisions
})

# Create a bar plot for accuracy per fold
sns.barplot(x=np.arange(1, len(accuracies)+1), y=accuracies)
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
conf_matrix_with_names = pd.DataFrame(overall_conf_matrix, index=class_names, columns=class_names) # Create a DataFrame for the confusion matrix


# Visualize the confusion matrix with Seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_with_names, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Class Names')
plt.show()