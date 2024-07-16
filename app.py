# Created on Tuesday, March 7, 2566 BE (GMT+7) Time in Suthep, Mueang Chiang Mai District, Chiang Mai
# @author: natthanaphop.isa
# @editor: wachiranun.sir
# Last update 15/7/2024

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image


# Loading the saved model
with open('logistic_model_frailty.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Loading banner and button image
image = Image.open('./images/app_banner.png')
st.image(image, use_column_width ="always" )

logo = './images/slidebar_button.png'



# Creating a function for Prediction
def frailty_prediction(input_data):
    prediction = loaded_model.predict(input_data)
    pred_prop = loaded_model.predict_proba(input_data)

    if prediction[0] == 0:
        pred = 'ท่านเป็นกลุ่มที่มีความเสี่ยง "ต่ำ" 🏃 ต่อการเกิดภาวะเปราะบางทางกายภาพ'
    else:
        pred =  'ท่านเป็นกลุ่มที่มีความเสี่ยง "สูง" 🧑🏼‍🦽 ต่อการเกิดภาวะเปราะบางทางกายภาพ แนะนำให้ท่านพบแพทย์เพื่อขอคำแนะนำในการเพิ่มการออกกำลังกาย และดูแลภาวะโภชนาการให้เหมาะสม'
    
    return pred, pred_prop

# sidebar for navigation
st.logo(logo, icon_image=logo)
with st.sidebar:
    
    selected = option_menu('เลือกโมเดลพยากรณ์',
                          
                          ['Frailty Classification Using Machine Learning Model'],
                          icons=['activity'],
                          default_index=0)
    
# Model using non-HDL cholestoral level
if (selected == "Frailty Classification Using Machine Learning Model"):
    
    def transform_input(AGE,SEX,STATUS,HT,DLP,BMI,waistcir,calfcir,exhaustion_choices):
        
        if SEX == 'ชาย':
            SEX = 1
        else:
            SEX = 0
    
        if STATUS == 'อาศัยอยู่ลำพัง':
            STATUS = 1
        else:
            STATUS = 0
        
        if DLP == 'ใช่':
            DLP = 1
        else:
            DLP = 0
    
        if HT == 'ใช่':
            HT = 1
        else:
            HT = 0
        
        if exhaustion_choices == 'ไม่ใช่':
            exhaustion_choices = 0
        else:
            exhaustion_choices = 1
            
            
        return [AGE,SEX,STATUS,HT,DLP,BMI,waistcir,calfcir,exhaustion_choices]
    
        
    with st.form('my_form'):
        st.write("ข้อมูลพื้นฐาน")
        AGE =  st.slider('อายุ (ปี)',60, 120)
        SEX = st.selectbox('เพศ', ['ชาย', 'หญิง'])    
        STATUS = st.selectbox('สถานะ', ['อาศัยอยู่ลำพัง', 'อาศัยอยู่กับครอบครัวหรือผู้อื่นๆ'])
        BMI =  st.slider('ดัชนีมวลกาย (kg/m2)',0, 80)
        waistcir =  st.slider('เส้นรอบเอว (cm)',0, 200)
        calfcir =  st.slider('เส้นรอบวงน่อง (cm)',0, 80)
        st.write("โรคประจำตัว")
        HT = st.selectbox('ได้รับการวินิจฉัยโรคความดันโลหิตสูง', ['ไม่ใช่', 'ใช่'])
        DLP = st.selectbox('ได้รับการวินิจฉัยโรคไขมันในเลือดสูง', ['ไม่ใช่', 'ใช่'])
        st.write("ความรู้สึกอ่อนล้าทางกาย")
        exhaustion_choices = st.selectbox('ท่านมักจะมีความรู้เหนื่อยล้าเป็นประจำใช่หรือไม่?', ['ไม่ใช่', 'รู้สึกเป็นบางครั้ง (1-2 วัน/สัปดาห์)','รู้สึกบ่อยครั้ง (3-4 วัน/สัปดาห์)', 'เป็นประจำทุกวัน'])

        if st.form_submit_button('พยากรณ์ความเสี่ยงของภาวะเปราะบางทางกายภาพ'):
            
            input_tranfrom = transform_input(AGE,SEX,STATUS,HT,DLP,BMI,waistcir,calfcir,exhaustion_choices)
            new_prediction = pd.DataFrame.from_dict({1:input_tranfrom}, columns=["AGE","SEX","STATUS","HT","DLP","BMI","waistcir","calfcir","exhaustion_choices"], orient='index')
            pred, pred_prop = frailty_prediction(new_prediction)
            pred_percent = pred_prop[-1,-1]*100
            pred_percent = format(pred_percent,".2f")
            pred
            text = "โอกาสท่านจะมีภาวะเปราะบางทางกายภาพ " + str(pred_percent) + " %"
            st.success(text, icon="🧑🏼‍🦽")

