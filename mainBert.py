#http://localhost:8501

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st
import pickle
import sklearn
import re
from transformers import BertTokenizer
import time
import smtplib
import re
from flask_mail import Mail, Message
import nltk
from nltk.corpus import stopwords
import tensorflow
from tensorflow import keras
#nltk.download('stopwords')

#model=pickle.load(open('Picklelr.pkl','rb'))
model = tensorflow.keras.models.load_model("Bert_for_nhs.h5", custom_objects={'TFBertMainLayer': TFAutoModel})

# from tensorflow.keras.utils import CustomObjectScope
#
# with CustomObjectScope({'TFBertMainLayer': TFBertMainLayer}):
#     model = load_model('Bert_for_nhs.h5')

#tokenizer = Tokenizer(num_words=100, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

st.title("Primary Care Network")

st.write("""
         Book an appointment
         """)
def analyse(Disease, Sym1, Sym2, Sym3, Sym4, Sym5, Sym6, Sym7, Sym8):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^5-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    symp = [Disease, Sym1, Sym2, Sym3, Sym4, Sym5, Sym6, Sym7, Sym8]
    dis = ''
    dis = (' '.join(str(x) for x in symp))
    text = dis.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub('nan', '', text)
    text = re.sub('_', ' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text

    tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
    tokens=tokenizer.encode_plus(text,max_length=512,
                                 truncation=True,padding='max_length',
                                 add_special_tokens=True,return_token_type_ids=False,
                                 return_tensors='tf')
    return {
        'input_ids': tensorflow.cast(tokens['input_ids'],tensorflow.float64),
        'attention_mask':tensorflow.cast(tokens['attention_mask'],tensorflow.float64)

    }

def send_mail(answer,email):
    ans = answer
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.ehlo()

    server.login('lasima.sn@gmail.com', 'exmjxsznufjsxdtd')

    subject = "Appointment booked!"
    body = "Your appointment has been booked with {dept} department at {time} ".format(dept=ans, time='2pm')
    msg = f"Subject: {subject}\n\n{body}"

    server.sendmail(
        'lasima.sn@gmail.com',
        email,
        msg
    )


with st.form("entry_form",clear_on_submit=False):
    #Disease,Sym1=st.columns(9)
    Disease= st.selectbox("Why are you here", ("Wrist , hand or finger pain","neck or back pain","pain in the leg","Head ache"))
    Sym1=st.text_input("Which part of your body is your concern associated with?")#, ("Wrist , hand or finger","neck or back","foot or ankle or leg","Head"))
    Sym2=st.selectbox("How long have you had your current symptoms?", ("less than 24 hrs","longer than 24 hrs","long term or intermittent"))
    Sym3=st.selectbox("Have you tried any of these medications?", ("Paracetamol","Aspirin or any other medication","Any cream or gel","Not tried"))
    #Sym4=st.selectbox("Have you been diagnosed with any long term medical problems?", ("Yes","Migrane or cluster or sinus headache","None"))
    Sym5=st.selectbox("Do you have any of the below symptoms?", ("No","Redness or Swelling or loss of sensation or painful lump","Change of shape or loss of movement"))
    Sym6=st.selectbox("Do you feel any of these difficulties accompanied with the pain?", ("No","fever lasting more than 5 days","Excessive sleepiness or change in concious level","Lack of energy","Jaw pain","Worsening headache"))
    Sym7=st.selectbox("Are you able to perform regular activities?", ("Can do regular activities","Unable to move affected area"))
    #Sym8=st.selectbox("Do you have any of these symptoms?",("None","Squint or eyes affected","Dizzy or associated with ataxia or unsteadiness","intense pain in bluish discolouration","persistent vomiting","stress","anxiety or depression"))
    Comment=st.text_area("Is there anything you want to tell us.",placeholder="Is there anything you want to tell us.")

    Email = st.text_area("Enter your email address",placeholder="Email address")
    "---"


    submitted=st.form_submit_button("Submit")
    if submitted:
        answer=analyse(Disease, Sym1, Sym2, Sym3, Sym4, Sym5, Sym6, Sym7, Sym8)
        dept=prediction(answer)
        st.write("You are being referred to {dept}. Click below to confirm your appointment.".format(dept=dept))

        confirmbtn=st.form_submit_button("Confirm")
        #send_mail(answer,email=Email)
        if confirmbtn:
            send_mail(answer)
#
def prediction(input):
    probs=model.predict(test)
    labels=['pharmacy', 'physio' ,'gp', 'podiatry','mental health','social councelling']
    return label[np.argmax(prob[0])]










