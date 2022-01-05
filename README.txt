#** SWBio DTP Machine Leanring Project - Scott Allen **

This project sets out to analyse the data from the Scikit Learn webpage, Breast
Cancer Wisconsin Diagnostic datset as part of the SWBio Data Science and Machine
Learning module. The data should download automatically using the code given
below. The code below can be copied directly into a Jupyter Notebook, 
but the Notebook itself is also found within the repository. 

Interpretation of the main findings are given in the assocaited report, Exploring 
the Breast Cancer Wisconsin Diagnostic Dataset.doc.

#Annotated code used in the project.
_____________________________________________________________________________

# Importing the various libraries/tools needed for my analysis.

import Bio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

fig=plt.figure()

_____

# Importing the Wisconsin Breast cancer dataset from Scikit learn.

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(dat.data, columns=dat.feature_names)

# No missing values are found within the data, therefore polishing/cleaning step is unnecessary.

df.info()

_____

# I added the target, malignant or benign, as it made it easier to visualise. 
#This was done by modifying code written by Chris Tran, chriskhanhtran.github.io. 

df_target = pd.DataFrame(data.target, columns=['target'])
dft = pd.concat([df, df_target], axis=1)
dft['target'] = dft['target'].apply(lambda x: "Benign"
                                  if x == 1 else "Malignant")

#Made a numerical dataframe, this is the split between malignant and benign as the 1 and 0.

Numerical_df = pd.DataFrame(np.c_[data['data'],data['target']],
             columns = np.append(data['feature_names'], ['target']))

dft.head()
#Numerical_df.head()

_____

#As the benign and malignant were divided as 0 or 1, I was able to split them into their own dataframes.
separator = 0.5
Malignant_df = Numerical_df[Numerical_df['target'] < separator]
Benign_df = Numerical_df[Numerical_df['target'] > separator]

Benign_df.head()
#Malignant_df.head()

_____

#Creates a graph to count the number of benign or malignant diagnoses.
sns.set_style('whitegrid')
dft['target'].value_counts()
sns.countplot(dft['target'], palette = 'Set1')
plt.xlabel('Diagnosis')
plt.title('Number of tumour types')
#plt.savefig('Count.png', dpi=300)

_____

#Heatmap of correlation values between variables in the malignant diagnoses.

Malignant_corr = Malignant_df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(Malignant_corr[Malignant_corr<0.75], vmin=-1.0, vmax=1.0, square=True, cmap='PiYG')
plt.title('Correlation of variable in Malignant cases')
plt.savefig('Mal_Corr.png', dpi=300)

_____

#Heatmap of correlation values between variables in the benign diagnoses.

Benign_corr = Benign_df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(Benign_corr[Benign_corr<0.75], vmin=-1.0, vmax=1.0, square=True, cmap='PiYG')
plt.title('Correlation of variable in Benign cases')
plt.savefig('Benign_corr.png', dpi=300)

_____

#Noting down interesting differences between relationships in the Malignant or Benign (done by eye),
#a pair plot was developed visualise this. 

Interesting_variables_plot = dft[['mean compactness','mean concave points','mean concavity','mean fractal dimension', 
                                  'mean radius','target']]
sns.pairplot(Interesting_variables_plot, hue='target', size = 4, kind="reg")
plt.savefig('interesting pairplots', dpi=300)

#specific relationships were then looked at in further detail, such as:

sns.relplot(data=dft, x="mean radius", y='mean concave points', hue="target")

_____

#A model was then trained.

X = Numerical_df.iloc[:, 1:30].values
Y = Numerical_df.iloc[:, 30].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

_____


# Logistic model.

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000000)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

#plots a confusion matrix.

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm,annot=True,cmap='PiYG')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('cm.png',dpi=300)

#tests accuracy of the model

accuracy_score(Y_test,Y_pred)

#my accuracy was 95.1%

_____


#Kneighbours.

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(X_train, Y_train)
Yn_pred = model.predict(X_test)

cm = confusion_matrix(Y_test, Yn_pred)
sns.heatmap(cm,annot=True,cmap='PiYG')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('cm.png',dpi=300)


accuracy_score(Y_test,Yn_pred)

#my accuracy was 93.7%











