# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




#The Trained File is load
Load = pd.read_csv('D:\Kaggle\cmd\train.csv')

print(Load.describe())
print(Load.head(), sep = '\n')




#The Prediction from the reports
Refer = pd.read_csv('D:\Kaggle\cmd\test.csv')

print(Refer.describe())
print(Refer.head(), sep = '\n')

#This carifying the features on the survived once
Gender_sub = pd.read_csv('D:\Kaggle\cmd\gender_submission.csv')

print(Gender_sub.describe())
print(Gender_sub.head(), sep = '\n')





#Here I am combining the train and test file to predict
Full_Data = [Load, Refer, Gender_sub]
print([Load, Refer, Gender_sub])





#First Phase
#Now onward the data will never check the Sex, PassengerId & Passenger_Name.
#It will pass the data which are exactly same at the test side.
for Each_Passenger in Full_Data :
    
    #Converterd to the integer dtype
    #Here the Gender taken and it reverify by the data
    #This steps is for inter check with train and test file
    Each_Passenger['passenger'] = Each_Passenger.Sex.str.extract('([A-Za-z]+)\. ', expand = False).apply(lambda x : x == Refer['PassengerId'] if type(x) == str else 1)
    Each_Passenger['passenger'] = pd.factorize(Each_Passenger['passenger'])[0]

    #And here it combine and coverted to one by overwrite
    #The person are unique and Sex can be same but not all of them are same
    Each_Passenger['passenger'] = Each_Passenger.Name.str.extract('([A-Za-z]+)\. ', expand = False)
    Each_Passenger['passenger'] = pd.factorize(Each_Passenger['passenger'])[0]
    
    print(Each_Passenger['passenger'])





#Second Phase
#Cleaning data and Gather match data 
#Mean Data Preprocessing
#This will gather a big catelog for the features
#The connection in the Fare with Age because Fare Age defur but
#In the category age and fare will have some similarity
for Each_Passenger in Full_Data :
    
    Each_Passenger.loc[(Each_Passenger['Age'] <= 0) & (Each_Passenger['Fare'] <= 0) , 'Age']  =  0
    Each_Passenger.loc[(Each_Passenger['Age'] > 0) & (Each_Passenger['Age'] <=6 ) | (Each_Passenger['Fare'] > 0) & (Each_Passenger['Fare'] <= 52) , 'Age'] = 1
    Each_Passenger.loc[(Each_Passenger['Age'] > 6) & (Each_Passenger['Age'] <=13 ) | (Each_Passenger['Fare'] > 52) & (Each_Passenger['Fare'] <= 126) , 'Age'] = 2
    Each_Passenger.loc[(Each_Passenger['Age'] > 13) & (Each_Passenger['Age'] <=22 ) | (Each_Passenger['Fare'] > 126) & (Each_Passenger['Fare'] <= 187) , 'Age'] = 3
    Each_Passenger.loc[(Each_Passenger['Age'] > 22) & (Each_Passenger['Age'] <=50 ) | (Each_Passenger['Fare'] > 187) & (Each_Passenger['Fare'] <= 279) , 'Age'] = 4
    Each_Passenger.loc[(Each_Passenger['Age'] > 80) & (Each_Passenger['Age'] <=80 ) | (Each_Passenger['Fare'] > 279) & (Each_Passenger['Fare'] <= 513) , 'Age'] = 5
    Each_Passenger.loc[(Each_Passenger['Age'] > 80) & (Each_Passenger['Fare'] > 513), 'Age'];   

    
print(Each_Passenger.describe())
print(Each_Passenger['Age'].tail(20))
