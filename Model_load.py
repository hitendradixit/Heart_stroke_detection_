import pickle
import numpy as np
import pandas as pd

def load_model(path):
    with open(path,'rb') as f:
        return pickle.load(f)


encoder=load_model("G:\Mini_project_new\Model\Label_encoder_1.pkl")        
dtc = load_model('G:\Mini_project_new\Model\stroke_prediction_model.pkl')


gender=input("Enter the gender: ")
age=int(input("Enter the age: "))
HT_yes=input("Enter the hypertension: ")
HD_yes=input("Enter the heart disease: ")
EM_yes=input("Enter the Married: ")
Urban=input("Enter the Urban: ")
G_level=float(input("Enter the Glucose level: "))
bmi=float(input("Enter the bmi: "))


sex = pd.Series(["male", "female"])
sex = encoder.fit_transform (sex)

ht= pd.Series(["yes","no"])
ht = encoder.fit_transform(ht)

hd=pd.Series(["yes","no"])
hd = encoder.fit_transform(hd)

em=pd.Series(["yes","no"])
em = encoder.fit_transform(em)

rs=pd.Series(["Urban","Rural"])
rs = encoder.fit_transform(rs)



    
if gender=="Male":
    gender=sex[0]
else:
    gender=sex[1]

    
if HT_yes=="yes":
    HT_yes=ht[0]
else:
     HT_yes=ht[1]

if HD_yes=="yes":
    HD_yes=hd[0]
else:
    HD_yes=hd[1]

if EM_yes=="yes":
    EM_yes=em[0]
else:
    EM_yes=em[1]

if Urban=="yes":
    Urban=rs[0]
else:
    Urban=rs[1]


data_main=np.array([[gender], [age], [HT_yes], [HD_yes], [EM_yes], [Urban], [G_level],[bmi]])
data_main=data_main.reshape(1,-1)
out = dtc.predict(data_main)[0]
if out==1:
    print("Will get a stroke")
else:
    print("Will not get a stroke")
