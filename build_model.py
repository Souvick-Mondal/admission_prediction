# Import required libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle


# Read data from admission.csv 
df = pd.read_csv("admission.csv")

# Consider only 3 features - Gre, Toefl, and Cgpa
# Chance is label 

x = df[['Gre','Toefl','Cgpa']]
y = df['Chance']

# Split data into train and test 
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Fit or Train the Model
lr_model = LinearRegression()
#lr_model.fit(X_train,y_train)
lr_model.fit(x, y)
pickle.dump(lr_model,open('model.pkl','wb'))

#model=pickle.load(open('model.pkl','rb'))
#print(model.predict([[322,110,8.67]]))

"""score = lr_model.score(X_train,y_train)
print(f"Accuracy with train data : {score:0.2f}")

# Evaluate Model using test data 
y_pred =lr_model.predict(X_test)

# Find out accuracy with test data 
r2score = r2_score(y_test,y_pred)
print(f"Accuracy with test data :  {r2score:0.2f}")

# Pickle model
pd.to_pickle(lr_model,'lr_model.pickle')
model = pd.read_pickle('lr_model.pickle') 
result = model.predict([[gre,tof,cgpa]])
print(f"Chances are : {result[0] * 100:.2f}%")
"""


