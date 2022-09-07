#http://localhost:8501
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import google_auth_oauthlib
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st
import pickle
import sklearn
import re
import time
import smtplib
import re
from flask_mail import Mail, Message
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow import keras
import sqlite3
import pandas as pd
import calndar as cal
import forMail as mailer
#nltk.download('stopwords')
conn=sqlite3.connect('data.db',check_same_thread=False,timeout=15)
curr=conn.cursor()
#op_df=table[int()].df
#st.dataframe(op_df)

#code to load and initialize models
model=pickle.load(open('teletubby.pkl','rb'))
#model = tf.keras.models.load_model("lstm_dataset.h5") #,custom_objects={'KerasLayer':hub.KerasLayer})
#tokenizer = Tokenizer(num_words=100)# filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#model = tf.saved_model.load(mobilenet_save_path)
st.title("Primary Care Network")

st.write("""
         Book an appointment
         """)

#code for adding to database
def addData(a,b,c,d):
    curr.execute("""CREATE TABLE IF NOT EXISTS nhs_form(NAME TEXT(50),DISEASE TEXT(50), EMAIL TEXT(50),DEPT TEXT(50));""")
    curr.execute("INSERT INTO nhs_form VALUES (?,?,?,?)",(a,b,c,d))
    conn.commit()
    conn.close()
    st.success("Successfully saved")


def analyse(Disease, Sym1, Sym2, Sym3, Sym5, Sym6, Sym7):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^5-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    symp = [Disease, Sym1, Sym2, Sym3, Sym5, Sym6, Sym7]
    dis = ''
    dis = (' '.join(str(x) for x in symp))
    text = dis.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub('nan', '', text)
    text = re.sub('_', ' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    #combined = analyse(Disease, Sym1, Sym2, Sym3, Sym4, Sym5, Sym6, Sym7, Sym8)
    #{tokenizer = Tokenizer(num_words=100, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, char_level=False)
    #tokenizer.fit_on_texts(text)
    #tokenized_text = tokenizer.texts_to_matrix(text)
    #pred = model.predict(tokenized_text)
    #return np.argmax(pred)
    #prediction = np.argmax(pred)}
    #tokenizer.fit_on_texts(text)
    # seq = tokenizer.texts_to_sequences([text])
    # padd = pad_sequences(seq, maxlen=20)
    # pred = model.predict(padd)
    # labels = ['gp', 'mental health', 'pharmacy', 'physio', 'podiatry', 'social councelling']
    # return labels[np.argmax(pred)]
    #result=int(np.argmax(pred))
    return model.predict([text])[0]
def send_mail(email):
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.ehlo()
    server.login('nhs.fortest@gmail.com', 'xjnafdjmegkhjwyz')
    subject = "Request received!"
    body = "Your appointment request has been received. We will contact you with details shortly.\nIn case of emergency call 999."
    msg = f"Subject: {subject}\n\n{body}"
    server.sendmail(
        'nhs.fortest@gmail.com',email,msg
    )

def send_appointment_confirmation(answer,email,appt_time,app_end):
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.ehlo()
    server.login('nhs.fortest@gmail.com', 'xjnafdjmegkhjwyz')
    subject = "Appointment booked!"
    body = "Your appointment has been booked with {dept} department from {st_time} to {en_time}. \nIn case you are not able to attend, please let us know. ".format(dept=answer, st_time=appt_time,en_time=app_end)
    msg = f"Subject: {subject}\n\n{body}"

    server.sendmail(
        'nhs.fortest@gmail.com', email, msg
    )

Name = st.text_input("Full Name")
page_names= ["Book a GP appointment or get health advice", "Refill Prescription", "Get help with Mental Health"]
Disease = st.selectbox("Why are you here", page_names)
print(Disease)
if (Disease == "Get help with Mental Health"):

    Sym1=st.text_area("What brings you here? Explain in a few words")
    Sym2 = st.selectbox("How long have you had your current condition?",
                        ("less than 24 hrs", "longer than 24 hrs", "long term or intermittent"))
    Sym3 = st.selectbox("Are you on any of these medications?",
                        ("Anti-depressants", "any other mood stabilizers", "Benzodiazepines", "No"))
    Sym5 = st.multiselect("Do you have any of the below symptoms?", (
        "No", "Anxiety", "Stress","Feeling low or depressed"))
    Sym6 = st.selectbox("Select whatever applies to you", (
        "No","Excessive sleepiness","Lack of energy","Avoiding social situations", "Worsening headache"))
    Sym7 = st.selectbox("Is this stopping you from perform regular activities?",
                        ("Doing regular activities as usual", "Little to no interest in doing activities"))

elif (Disease == "Book a GP appointment or get health advice"):

    Sym1=st.text_area("What brings you here? Explain in a few words")
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
    Sym1 = st.text_area("What brings you here? Explain in a few words")
    Sym2 = st.selectbox("Funny how How long have you had your current symptoms?",
                        ("less than 24 hrs", "longer than 24 hrs", "long term or intermittent"))
    Sym3 = st.selectbox("Have you tried any of these medications?",
                        ("Paracetamol", "Aspirin or any other medication", "Any cream or gel", "Not tried"))
    # Sym4=st.selectbox("Have you been diagnosed with any long term medical problems?", ("Yes","Migrane or cluster or sinus headache","None"))
    Sym5 = st.selectbox("Do you have any of the below symptoms?", (
        "No", "Redness or Swelling", "loss of sensation", "painful lump", "Change of shape ", " loss of movement"))
    Sym6 = st.selectbox("Do you feel any of these difficulties accompanied with the pain?", (
        "No", "fever lasting more than 5 days", "Excessive sleepiness or change in concious level",
        "Lack of energy","Jaw pain", "Worsening headache"))
    Sym7 = st.selectbox("Are you able to perform regular activities?",
                        ("Can do regular activities", "Unable to move affected area"))
    # Sym8=st.selectbox("Do you have any of these symptoms?",("None","Squint or eyes affected","Dizzy or associated with ataxia or unsteadiness","intense pain in bluish discolouration","persistent vomiting","stress","anxiety or depression"))
print(Sym5)
#Comment = st.text_input("Is there anything you want to tell us.",
#                    placeholder="Is there anything you want to tell us.")
Email = st.text_input("Enter your email address", placeholder="Email address")
"---"

submitted=st.button("Submit")
if submitted:
    send_mail(Email)
    answer=analyse(Disease, Sym1, Sym2, Sym3, Sym5, Sym6, Sym7)
    st.write("You are being referred to {dept}. We will get back to you shortly with your appointmnet details.".format(dept=answer))
    st.success("Appointment request received")

    #code to save to sqllite
    addData(Name,Disease,Email,answer)

    #code to get calendar events
    service,events,start,end,duration = cal.get_calendar_events()
    appt_time,app_end=cal.insert_new_calendar_event(service,Name, Sym1,Email,events, start, end, duration)
    send_appointment_confirmation(answer,Email,appt_time,app_end)
        #code to convert df to csv
        #df_new = pd.DataFrame({'Why are you here': [Disease],
                          #   'Which part of your body is affected?': [Sym1],
                        #   'Department': [answer],









#                               })
        #df=pd.DataFrame()
        #df=pd.concat([df,df_new],ignore_index=True)
        #st.dataframe(st.session_state.df)
        #results=pd.DataFrame(answer,columns=['Ans'])
        #st.dataframe(results)
#st.download_button(label="Download",data=df.to_csv(),mime='text/csv')

        #

    #if submitted not in st.session_state:
    #    st.session_state.submitted=False
    #def callback():
    #    st.session_state.submitted=True

    #if(
    #    st.form_submit_button("Submit",on_click=callback())

    #):
     #   if not st.form_submit_button("Confirm"):
     #       pass
      #  else:
       #     st.send_mail(answer)
#


        #df_new = pd.DataFrame({'Why are you here': [Disease],
        #                        'Which part of your body is affected?': [Sym1],
        #                        'Email': [Email],
        #                        'Department': [answer],
        #
        #                        })
        # df = pd.DataFrame(data=df_new)
        #
        # st.write(df)
        # open('df.csv','w',write(df.to_csv()))
        # #st.session_state.df=pd.append([st.session_state.df,df_new],axis=0)
        # #st.dataframe(st.session_state.df)
        # st.download_button("Export",
        #                    df.to_csv(),
        #                    mimi='text/csv')
        # #confirmbtn=st.form_submit_button("Confirm")
        # #send_mail(answer,email=Email)
        # #if confirmbtn:
        # #send_mail(answer)








