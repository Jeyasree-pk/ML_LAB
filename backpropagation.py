import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
def sigmoid(x):
   return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
   return x*(1-x)
x,y=make_classification(n_samples=500,n_features=4,n_informative=3,n_redundant=0,n_classes=2,random_state=42)
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=y.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
input_size=x.shape[1]
hidden_size=input_size//2
output_size=1
# w1=np.random.rand(input_size,hidden_size)-1
# w2=np.random.rand(hidden_size,output_size)-1
w1=[]
for i in range(input_size):
 w=input(f"Enter {hidden_size} weight for {i+1} neuron ")
 w1.append([float(l) for l in w.split()])
w1=np.array(w1)
w2=[]
for i in range(hidden_size):
  o=input(f"Enter weight for hidden {i+1}" ) 
  w2.append([float(p) for p in o.split()])
w2=np.array(w2)
lr=0.1
epochs=10000
loss_history=[]
for epoch in range(epochs):
  hidden_input=np.dot(x_train,w1)
  hidden_output=sigmoid(hidden_input)
  final_input=np.dot(hidden_output,w2)
  final_output=sigmoid(final_input)
  error=y_train-final_output
  loss=np.mean(np.square(error))
  loss_history.append(loss)

  d_final=error*sigmoid_derivative(final_output)
  error_hidden=d_final.dot(w2.T)
  d_hidden=error_hidden*sigmoid_derivative(hidden_output)
  w1+=x_train.T.dot(d_hidden)*lr
  w2+=hidden_output.T.dot(d_final)*lr
  if epoch%1000==0:
    print(f"epoch {epoch}-loss {loss} ")
hidden_out_test=sigmoid(np.dot(x_test,w1))
final_out_test=sigmoid(np.dot(hidden_out_test,w2))
y_pred=(final_out_test>0.5).astype(int)    
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy: {accuracy}")
plt.figure(figsize=(10,8))
plt.plot(loss_history,label="Training data")
plt.xlabel('Epochs')
plt.ylabel('Mean squared error loss')
plt.title('Traing loss over epochs')
plt.legend()
plt.show()
