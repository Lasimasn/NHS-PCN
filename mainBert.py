#http://localhost:8501

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st
import pickle
import sklearn
import re
import transformers
from transformers import BertTokenizer, TFBertMainLayer
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
model = tensorflow.keras.models.load_model("Bert_nhs_sheet4.h5", custom_objects={'CustomMetric':transformers.TFBertMainLayer})

# from tensorflow.keras.utils import CustomObjectScope
#
# with CustomObjectScope({'TFBertMainLayer': TFBertMainLayer}):
#     model = load_model('Bert_for_nhs.h5')

#tokenizer = Tokenizer(num_words=100, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

st.title("Primary Care Network")

st.write("""
         Book an appointment
         """)
def analyse(Disease, Sym1, Sym2, Sym3,  Sym5, Sym6, Sym7):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^5-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    symp = [Disease, Sym1, Sym2, Sym3,  Sym5, Sym6, Sym7]
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
def final_output(input):
    probs=model.predict(input)
    labels=['pharmacy', 'physio' ,'gp', 'podiatry','mental health','social councelling']
    return labels[np.argmax(probs[0])]

# with st.form("entry_form",clear_on_submit=False):
Name = st.text_input("Full Name")
page_names= ["Book a GP appointment or get health advice", "Refill Prescription", "Get help with Mental Health"]
Disease = st.selectbox("Why are you here", page_names)
print(Disease)
if (Disease == "Get help with Mental Health"):

    # Sym1=st.selectbox("Which part of your body is your concern associated with?", ("Wrist , hand or finger","neck or back","foot or ankle or leg","Head"))
    Sym2 = st.selectbox("How long have you had your current condition?",
                        ("less than 24 hrs", "longer than 24 hrs", "long term or intermittent"))
    Sym3 = st.selectbox("Have you tried any of these medications?",
                        ("Paracetamol", "Aspirin or any other medication", "Any cream or gel", "Not tried"))
    Sym5 = st.multiselect("Do you have any of the below symptoms?", (
        "No", "Little to no interest in doing activities", "Excessive sleepiness",
        "change in concious level",
        "Lack of energy",))
    Sym6 = st.selectbox("Do you feel any of these difficulties accompanied with the pain?", (
        "No", "Little to no interest in doing activities", "Excessive sleepiness",
        "change in concious level",
        "Lack of energy",
        "Anxiety in social situations", "Worsening headache"))
    Sym7 = st.selectbox("Is this stopping you from perform regular activities?",
                        ("Can do regular activities", "Unable to do regular activities"))

elif (Disease == "Book a GP appointment or get health advice"):

    # Sym1=st.selectbox("Which part of your body is your concern associated with?", ("Wrist , hand or finger","neck or back","foot or ankle or leg","Head"))
    Sym2 = st.selectbox("How long have you had your current symptoms?",
                        ("less than 24 hrs", "longer than 24 hrs", "long term or intermittent"))
    Sym3 = st.selectbox("Have you tried any of these medications?",
                        ("Paracetamol", "Aspirin or any other medication", "Any cream or gel", "Not tried"))
    # Sym4=st.selectbox("Have you been diagnosed with any long term medical problems?", ("Yes","Migrane or cluster or sinus headache","None"))
    Sym5 = st.selectbox("Do you have any of the below symptoms?", (
        "No", "Redness or Swelling", "loss of sensation", "painful lump", "Change of shape ", " loss of movement"))
    Sym6 = st.selectbox("Do you feel any of these difficulties accompanied with the pain?", (
        "No", "fever lasting more than 5 days", "Excessive sleepiness or change in concious level",
        "Lack of energy",
        "Jaw pain", "Worsening headache"))
    Sym7 = st.selectbox("Are you able to perform regular activities?",
                        ("Can do regular activities", "Unable to move affected area"))
else:

    Sym2 = st.selectbox("Funny how How long have you had your current symptoms?",
                        ("less than 24 hrs", "longer than 24 hrs", "long term or intermittent"))
    Sym3 = st.selectbox("Have you tried any of these medications?",
                        ("Paracetamol", "Aspirin or any other medication", "Any cream or gel", "Not tried"))
    # Sym4=st.selectbox("Have you been diagnosed with any long term medical problems?", ("Yes","Migrane or cluster or sinus headache","None"))
    Sym5 = st.selectbox("Do you have any of the below symptoms?", (
        "No", "Redness or Swelling", "loss of sensation", "painful lump", "Change of shape ", " loss of movement"))
    Sym6 = st.selectbox("Do you feel any of these difficulties accompanied with the pain?", (
        "No", "fever lasting more than 5 days", "Excessive sleepiness or change in concious level",
        "Lack of energy",
        "Jaw pain", "Worsening headache"))
    Sym7 = st.selectbox("Are you able to perform regular activities?",
                        ("Can do regular activities", "Unable to move affected area"))
    # Sym8=st.selectbox("Do you have any of these symptoms?",("None","Squint or eyes affected","Dizzy or associated with ataxia or unsteadiness","intense pain in bluish discolouration","persistent vomiting","stress","anxiety or depression"))
print(Sym5)
Comment = st.text_input("Is there anything you want to tell us.",
                    placeholder="Is there anything you want to tell us.")
Email = st.text_input("Enter your email address", placeholder="Email address")
"---"
sumbitted=st.button("Submit")
if sumbitted:
    answer=analyse(Disease,Sym1,Sym2,Sym3,Sym5,Sym6,Sym7)
    dept=final_output(answer)
    st.write("You are being referred to {dept}. Click below to confirm your appointment.".format(dept=dept))
    # submitted=st.form_submit_button("Submit")
    # if submitted:
    #     answer=analyse(Disease, Sym1, Sym2, Sym3,  Sym5, Sym6, Sym7)
    #     dept=final_output(answer)
    #     st.write("You are being referred to {dept}. Click below to confirm your appointment.".format(dept=dept))

#
# def get_conig(self):
#     return {'units':self.units}









