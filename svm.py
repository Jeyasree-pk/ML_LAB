import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
data=pd.read_csv("ml/code/svm.csv")
x=data.iloc[:,:-1]
x=pd.get_dummies(x)
Y=data.iloc[:,-1]
print(x.head())
print(Y.head())
le=LabelEncoder()
y=le.fit_transform(Y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
svm=SVC(kernel='linear')
svm.fit(x_train,y_train)
y_predict=svm.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
conf_matrix=confusion_matrix(y_test,y_predict)
disp=ConfusionMatrixDisplay(conf_matrix,display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion matrix")
plt.show()
pca=PCA(n_components=2)
x_reduced=pca.fit_transform(x_train)
svm_reduced=SVC(kernel='linear')
svm_reduced=svm.fit(x_reduced,y_train)
xx,yy =np.meshgrid(np.linspace(x_reduced[:,0].min()-1,x_reduced[:,0].max()+1,100),
                 np.linspace(x_reduced[:,1].min()-1,x_reduced[:,1].max()+1,100))
X_test_reduced = pca.transform(x_test)       # transform test data using same PCA
z=svm_reduced.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure(figsize=(10,8))
plt.contourf(xx,yy,z,alpha=0.8,cmap='coolwarm')
plt.scatter(x_reduced[:,0],x_reduced[:,1],c=y_train,edgecolors='k',marker='o')
plt.legend()
plt.show()
