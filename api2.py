# Import flask and datetime module for showing date and time
from flask import Flask
#Importing Pandas Library 
import pandas as pd
import numpy as np
#Importing train_test_split For Spliting Our Train Data To 80-20%
from sklearn.model_selection import train_test_split
#Importing VarianceThreshold  From Features Selection
from sklearn.ensemble import RandomForestClassifier
import requests
from sklearn.linear_model import RidgeClassifier
from io import StringIO
app = Flask(__name__)
@app.route('/getdata/<int:temp>/<int:heart>')
def getdata(temp,heart):
    URL = "https://drive.google.com/file/d/1AqmZVwNwjpfv-El7CIn8_fae9C7BFYxe/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
    train = pd.read_csv(path)



    data = {'Temperature':[temp],
        'Heart':[heart]
        
        }

    test = pd.DataFrame(data,columns=['Temperature','Heart'])

#Firstly Training 3 Models For Train Data(Without Normalization)
#Spliting Our Train Data(Without Normalization) Into 80% And 20%

#For Labels
    Labels = train.Class

#For Features
    Features = train.drop('Class', axis=1)

#Splitting The Data Into 80%(For Training t_train) And 20%(For Testing t_test)
    t_train, t_test, y_train, y_test = train_test_split(Features, Labels,test_size=0.2)
#Here We Are #Fitting Our training Data
    clf = RidgeClassifier().fit(t_train, y_train)
#Predicting The Values Of Local Splitted Test Data
    Cover_type = clf.predict(t_test)
#Printing The Predicted Value Of Local Splitted Test Data
    print("The Predicted Values Cover Type Of Local Test Data",Cover_type)
#Checking The  accuracy This Model Using The Local Test Data 
    Ridge=clf.score(t_test,y_test)
#Printing The Accuracy Score Of The Decision Tree Classifier 
    print("The Accuracy Score Of RidgeClassifier On Local Test Data",Ridge*100)

    Class2 = clf.predict(test)
    print("The Predicted Values  Class Of Local Test Data",Class2)
    py_list = Class2.tolist()
    c = py_list[0]


    
    return {
        'Temperature':temp, 
        "Heart":heart,
        "Class" : c,
        
        }
  
# Initializing flask app

  
  
# Route for seeing a data


  
    # Returning an api for showing in  reactjs
  
  
      
# Running app
if __name__ == '__main__':
      app.run(host="127.0.0.1", port=5000, debug=True)