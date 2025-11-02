import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv(r'C:\Users\jeyas\Desktop\lab\ml\code\iris.csv') 
X = iris.iloc[:,:-1]
y = iris.iloc[:,-1] 
X=pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
accuracy_dtc = metrics.accuracy_score(y_test, y_pred_dtc)
precision_dtc = metrics.precision_score(y_test, y_pred_dtc, average='weighted')
recall_dtc = metrics.recall_score(y_test, y_pred_dtc, average='weighted')

print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_dtc)
print("Precision:", precision_dtc)
print("Recall:", recall_dtc)
 
plot_tree(
    dtc,
    feature_names=X.columns,      # use DataFrame column names
    class_names=y.unique(),       # extract unique class labels
    filled=True
)

plt.show()


