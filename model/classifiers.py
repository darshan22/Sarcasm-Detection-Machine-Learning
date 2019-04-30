"""
This script performs the maodel training and testing. The tasks performed in this
script are finding correlation between features, feature importance using random forest classifier and training of SVM classifier.
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
plt.style.use('bmh')


#read the data into a dataframe
data = pd.read_csv("../Data/final_dataset.csv")

#finding the correlation matrix
columns = ['Emoji', 'Capital Words', 'User Mentions', 'Hashtags', 'Slang laughter Exp', 'Punctuations', 
          '+ve Words', '-ve words', 'neutral wors', 'Polarity', 'Polarity flip']

plt.figure(figsize=(12,8))
cm = np.corrcoef(data[columns].values.T)
hm = sns.heatmap(cm,
                 cmap = "RdBu_r",
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=columns,
                 xticklabels=columns)

plt.xticks(rotation = 45, fontsize = 12)
plt.yticks(rotation = 45, fontsize = 12)
plt.title("Correlation plot (left)")
plt.savefig('correlation_map.jpg')
# Show heat map
plt.show()

#build train and test set
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

#scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#function definition for random forest classifier.
def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=0) #Create an instance of classifier
    clf.fit(X_train, y_train) #fit the classifier to Train data
    feature_importances = clf.feature_importances_ #retrieve feature importances
    
    plot_importances = pd.Series(feature_importances, data.columns[:-1])
    plot_importances.sort_values(ascending = False, inplace = True) #sort the feature importance in descending order
    plt.figure(figsize = (12,8), dpi = 250)
    plot_importances.plot(x='Features', y='Importance', kind = 'bar', rot = 45)    #plot feature importance
    plt.xticks(rotation='vertical')
    #plt.savefig("feature_importances.png")
    plt.tight_layout()
    plt.savefig('feature_importance.jpg')
    plt.show()
    
    y_pred = clf.predict(X_test) #predict on test data
    
    #calculate precision, recall, fscore and support
    tp,fp,tn,fn=0,0,0,0
    for predicted,actual in zip(y_pred,y_test):
        if(predicted==actual and predicted==1): tp+=1
        elif(predicted==actual and predicted==0):tn+=1
        elif(predicted==0):fn+=1
        else:fp+=1
    
    accuracy = round(accuracy_score(y_test, y_pred),2)
    precision = round(float(tp)/(tp+fp),2)
    recall = round(float(tp)/(tp+fn),2)
    fscore = round(2*precision*recall/(precision+recall),2) 
    return accuracy,precision,recall,fscore


#define svm classifier
def svm_classifier(X_train, X_test, y_train, y_test):
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test_std)
    
    tp,fp,tn,fn=0,0,0,0
    for predicted,actual in zip(y_pred,y_test):
        if(predicted==actual and predicted==1): tp+=1
        elif(predicted==actual and predicted==0):tn+=1
        elif(predicted==0):fn+=1
        else:fp+=1
    
    accuracy = round(accuracy_score(y_test, y_pred),2)
    precision = round(float(tp)/(tp+fp),2)
    recall = round(float(tp)/(tp+fn),2)
    fscore = round(2*precision*recall/(precision+recall),2)
    
    return accuracy, precision, recall, fscore

#define a neural network
def neural_network(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(output_dim = 8, input_dim=X_train.shape[1], init='uniform', activation='relu'))
    model.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=25, batch_size=64)
    scores = model.evaluate(X_train, y_train)
    y_pred = model.predict(X_test)
    rounded = [int(round(x[0])) for x in y_pred]
    tp,fp,tn,fn=0,0,0,0
    for predicted,actual in zip(rounded,y_test):
        if(predicted==actual and predicted==1): tp+=1
        elif(predicted==actual and predicted==0):tn+=1
        elif(predicted==0):fn+=1
        else:fp+=1
    
    accuracy = round(scores[1], 2)
    precision = round(float(tp)/(tp+fp),2)
    recall = round(float(tp)/(tp+fn),2)
    fscore = round(2*precision*recall/(precision+recall),2)
    
    return accuracy, precision, recall, fscore

print("Training Random Forest classifier")
acc_rf, precision_rf, recall_rf, fscore_rf = random_forest(X_train_std, X_test_std, y_train, y_test)

print("Training SVM classifier")
acc_svm, precision_svm, recall_svm, fscore_svm = svm_classifier(X_train_std, X_test_std, y_train, y_test)

print("Training Neural Network")
acc_nn, precision_nn, recall_nn, fscore_nn = neural_network(X_train_std, X_test_std, y_train, y_test)
