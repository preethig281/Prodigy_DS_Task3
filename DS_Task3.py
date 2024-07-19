import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from  sklearn.metrics import accuracy_score

data = pd.read_csv(r"C:\Users\VENKATESH\OneDrive\Desktop\project\bank.csv")

data=pd.DataFrame(data)


print(data.columns)

encode = LabelEncoder()

hot= OneHotEncoder()



#data['y']=encode.fit_transform(data['y'])

#data[['job','marital','education','default','housing','loan','contact','month','poutcome']] = encode.fit_transform(data[['job','marital','education','default','housing','loan','contact','month','poutcome']])




model= DecisionTreeClassifier()
x=data.iloc[:,0:16]
x=hot.fit_transform(x)
y=data["y"]
y=encode.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4)

model =DecisionTreeClassifier()

model.fit(x_train,y_train)

print(x_train)

pred = model.predict(x_test)

accuracy = accuracy_score(pred, y_test)
print(accuracy)

