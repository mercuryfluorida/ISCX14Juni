from __future__ import division 
from sklearn import svm  
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  
  
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import roc_curve, auc 

 


import itertools
import numpy as np
import matplotlib.pyplot as plt
 
import pandas as pd 

print(__doc__) 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',  cmap=plt.cm.Blues ):
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
 

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    s = 'True label : ' , labels
    plt.ylabel(s)
    s = 'Predicted label : ' , labels
    plt.xlabel(s)

train_dataframe = pd.read_csv('ISCX14Juni.csv', header = 0, delimiter = ";")
train_dataframe = train_dataframe.dropna(how='any') 
train_dataframe = train_dataframe[0:50000] 
   
train_data = train_dataframe[['totalSourceBytes',
     'totalDestinationBytes',
     'totalDestinationPackets',
     'totalSourcePackets',
     'direction',
     'sourceTCPFlagsDescription',
     'destinationTCPFlagsDescription', 
     'protocolName',
     'sourcePort',
     'destination',
     'destinationPort',
     'startDateTime',
     'stopDateTime' ]]
train_data = train_data.drop(train_data.columns[[9, 11, 12]], axis=1)  
 

train_label_class  = train_dataframe.Tag
labels = list(set(train_label_class))
train_labels = np.array([labels.index(x) for x in train_label_class])
 
X  = train_data
y_ = train_labels
class_names = train_label_class 

 
X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=.4,
                                                    random_state=0)
   
classifier =  svm.SVC(kernel='poly', probability=True ) 
y_pred = classifier.fit(X_train, y_train).predict(X_test);  

cnf_matrix  = confusion_matrix(y_test, y_pred) 
np.set_printoptions(precision=2) 
 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='SVM polynomial Confusion matrix, without normalization' )
 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='SVM polynomial Normalized confusion matrix' )

plt.show()
  

results = classifier.predict(train_data)
num_correct = (results == train_labels).sum()
recall = 0.0
recall = num_correct / len(train_labels)
print ("SVM polynomial model accuracy (%): ", recall * 100, "%")  

 

X  = train_data
y  = train_labels 
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape
 
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
 
cv = StratifiedKFold(n_splits=2) 

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
    
    
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM polynomial Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


 
