# coding: utf-8

# ### dataset reformating

# In[2]:


import numpy as np
import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('data.csv', sep = '^', quoting=csv.QUOTE_NONE, encoding='utf-8', header=0)
count=Counter(line[0] for line in dataset['party'])
print(count)
plt.figure(1)
plt.bar(count.keys(),count.values())
plt.title('Number of democratic and republican tweets')


# In[5]:


dataset.info()


# ### data cleaning

# In[6]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[7]:


corpus = []
for i in range(0, 11543):
    review = re.sub('[^a-zA-Z]', ' ', dataset['content'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# ### feature extraction

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X=cv.fit_transform(corpus).toarray()
y=dataset['party'].values


# ### data splitting method 1 choose 4th every 5 sentences

# In[9]:


index = []
for ii in range(0, 11543):
    index.append(ii)
rIndex = index[4::5]

X_test = X[4::5]
X_train = X 
X_train = np.delete(X_train, rIndex, axis = 0)
y_test = y[4::5]
y_train = y 
y_train = np.delete(y_train, rIndex)


# In[11]:


print(len(X_test), len(X_train), len(y_test), len(y_train))
y_test=y_test.reshape((2308,1))


# ### classification and evaluation
# ### 1. Random Forest

# In[12]:


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc


# In[13]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
y_predict=rf_classifier.predict(X_train)
y_pred_rf = rf_classifier.predict(X_test)
print(y_test.shape)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_train,y_predict))
print(classification_report(y_test, y_pred_rf))

ytest,ypredrf=np.zeros((2308,1)),np.zeros((2308,1))
ytest[y_test=='Democratic Party']=0
ytest[y_test=='Republican Party']=1
rfscore=rf_classifier.fit(X_train, y_train).predict_proba(X_test) 
fpr,tpr,threshold= roc_curve(ytest,rfscore[:,1])
roc_auc=auc(fpr,tpr)
plt.title('ROC of Random Forest classifier')
plt.plot(fpr, tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### 2. Naive Bayes

# In[14]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_predict=nb_classifier.predict(X_train)
y_pred_nb = nb_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_train,y_predict))
print(classification_report(y_test, y_pred_nb))

nbscore=nb_classifier.fit(X_train, y_train).predict_proba(X_test) 
fpr,tpr,threshold= roc_curve(ytest,nbscore[:,1])
roc_auc=auc(fpr,tpr)
plt.title('ROC of Naive Bayes classifier')
plt.plot(fpr, tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### 3. MLP

# In[16]:


from sklearn.neural_network import MLPClassifier
nn_classifier = MLPClassifier(hidden_layer_sizes = (600,2), activation='relu', solver='adam', verbose=True, learning_rate = 'constant', max_iter=500)
nn_classifier.fit(X_train, y_train)
y_pred_nn = nn_classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))


# In[274]:


nnscore=nn_classifier.fit(X_train, y_train).predict_proba(X_test) 
fpr,tpr,threshold= roc_curve(ytest,nnscore[:,1])
roc_auc=auc(fpr,tpr)
plt.title('ROC of Neural Network classifier')
plt.plot(fpr, tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### 4. KNN

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y0_pred=knn.predict(X_test)
y0_predict=knn.predict(X_train)
print(confusion_matrix(y_test, y0_pred))
print(classification_report(y_train,y0_predict))
print(classification_report(y_test,y0_pred))

knnscore=knn.fit(X_train, y_train).predict_proba(X_test) 
fpr,tpr,threshold= roc_curve(ytest,knnscore[:,1])
roc_auc=auc(fpr,tpr)
plt.title('ROC of KNN classifier')
plt.plot(fpr, tpr,'b',label='AUC=%0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### 5. SVM

# In[122]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
score='recall'
print("# Tuning hyper-parameters for %s" % score)
clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                   scoring='%s_macro' % score)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))


# ### 6. Pipeline in TF-IDF

# In[18]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
pipeline = Pipeline([('vectorizer', CountVectorizer()),('tfidf', TfidfTransformer()), ('mlp', MLPClassifier(hidden_layer_sizes=(100,2)))])


# In[19]:


x1 = dataset['content'][:11544]
y1 = dataset['party'][:11544]
x1[:5]


# ### data splitting method 2 pick 20% randomly

# In[20]:


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)


# In[21]:


pipeline.fit(x1_train, y1_train)


# In[22]:


predictions = pipeline.predict(x1_test)
print(classification_report(y1_test, predictions))

