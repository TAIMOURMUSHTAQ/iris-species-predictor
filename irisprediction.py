#Importing Libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix

#Loadind Iris Dataset
data_set=pd.read_csv("C:\\Users\\LENOVO\\Downloads\\archive (7)\\Iris.csv")

#Encoding species labels
species_encoder=LabelEncoder()
data_set['Species']=species_encoder.fit_transform(data_set['Species'])

#Features and Targets Defining
X=data_set[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=data_set['Species']

#Training and Testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Model Training
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#Making Predictions
y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)

#Using TKAgg backend for VS IDE
matplotlib.use('TkAgg')

#Generating Confusion Matrix
cm=confusion_matrix(y_test,y_pred)

#Ploting and Saving
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.savefig("Confusion_Matrix.png")
print("Confusion Matrix saved as 'Confusion_Matric.png'")

#Streamlit UI 
st.title("🌸Iris Flower Species Predictions")
st.write(f"🎯Model Accuracy:**{acc*100:.2f}%**")
st.write("Enter Flower measurments below to predict it's Species:")
#User Input
sepal_length=st.number_input("Sepal Length (cm)",min_value=0.0,value=5.1)
sepal_width=st.number_input("Sepal Width (cm)",min_value=0.0,value=3.5)
petal_length=st.number_input("Petal Length (cm)",min_value=0.0,value=1.4)
petal_width=st.number_input("Petal Width (cm)",min_value=0.0,value=0.2)

if st.button("Predict Species"):
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction=model.predict(input_data)
    species=species_encoder.inverse_transform(prediction)
    st.success(f"🌼Predicted Species:**{species[0]}**")