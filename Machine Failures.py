#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[5]:


df=pd.read_csv("/Users/Hari's Mac/Downloads/F1/train.csv")


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(columns=["id","Product ID"],axis=1,inplace=True)


# In[9]:


df.head()


# In[10]:


df["Machine failure"].unique()


# In[11]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Type"]=le.fit_transform(df["Type"])


# In[12]:


df.head()


# In[13]:


x=df.drop(columns="Machine failure",axis=1)
x.head()


# In[14]:


y=df["Machine failure"]
y[0:5]


# In[15]:


y.unique()


# In[16]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape


# In[17]:


#ANN CLASSIFICATION
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[18]:


model=Sequential()
#adding input layer
model.add(Dense(11,activation="relu"))
#adding hidden layers
model.add(Dense(1600,activation="relu"))
model.add(Dense(200,activation="relu"))
#adding output layer
model.add(Dense(2,activation="softmax"))


# In[19]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[20]:


model.fit(xtrain,ytrain,epochs=10,batch_size=5,validation_data=(xtest,ytest))


# In[21]:


ypred=model.predict(xtest)


# In[22]:


ypred=model.predict([[1,300.6,309.6,1596,36.1,140,0,0,0,0,0]])
ypred = np.argmax(ypred)
output = [0,1]
output[ypred]


# In[23]:


ypred_1=model.predict([[0,302.3,311.5,1499,38.0,60,0,1,1,0,1]])
ypred = np.argmax(ypred)
output = [0,1]
output[ypred]


# In[27]:


import numpy as np

value = np.int64(42)
print(value)


# In[31]:



import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

# Load your trained model here
# Example:
model = keras.models.load_model('model1.h5')

# Function to make predictions
def predict(machine_params):
    # Preprocess the input data (machine_params) if needed
    # Example:
    # machine_params = preprocess_input(machine_params)

    # Make predictions using your model
    predictions = model.predict(machine_params)

    # Return the predicted class (or probabilities, if needed)
    return predictions

# Streamlit UI
st.title('Machine Failure Prediction')

# Create input fields for machine parameters
st.write('Enter Machine Parameters:')
param1 = st.number_input('Parameter 1', min_value=0.0, max_value=1000.0)
param2 = st.number_input('Parameter 2', min_value=0.0, max_value=1000.0)
# Add more input fields for other parameters as needed

# Create a button to trigger predictions
if st.button('Predict'):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'Parameter 1': [param1], 'Parameter 2': [param2]})
    # Call the predict function with the input data
    input_data = input_data.astype('float32')  # Convert to float32

    predictions = predict(input_data)
    
    # Display the prediction results
    st.write('Prediction Results:')
    st.write(predictions)

# Optionally, you can add more features like displaying prediction probabilities, explanations, etc.

