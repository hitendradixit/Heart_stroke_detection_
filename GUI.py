import streamlit as st
import pickle
import numpy as np
import pandas as pd


st.title("Heart Stroke Predictor")

select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
if not st.sidebar.checkbox("Hide", True, key='1'):

    def load_model(path):
        with open(path,'rb') as f:
            return pickle.load(f)
        
    with st.spinner('Please wait'):
        
        encoder=load_model("G:\Mini_project_new\Model\Label_encoder_1.pkl") 
        dtc = load_model('G:\Mini_project_new\Model\stroke_prediction_model.pkl')
        st.success('model is loaded')

    st.header('Enter the  details')
    name = st.text_input("Name:")
    gender=st.text_input(label="Gender",value="Male")
    age=st.number_input(label="Age")
    HT_yes=st.text_input(label="Hypertension")
    HD_yes=st.text_input(label="Heart Disease")
    EM_yes=st.text_input(label="Ever Married")
    Urban=st.text_input(label="Urban Residence")
    G_level=st.number_input(label="G_level",value=0)
    bmi=st.number_input(label="BMI",value=21)

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

    btn=st.button("Get Prediction")



    if btn:
        try:
            data=np.array([[gender], [age], [HT_yes], [HD_yes], [EM_yes], [Urban], [G_level],[bmi]])
            data=data.reshape(1,-1)
            result =dtc.predict(data)[0]
            st.success(result)
            if result==1:
                st.write("Will get a stroke")
            else:
                st.write("Will not get a stroke")
        except Exception as e:
            st.error(e)