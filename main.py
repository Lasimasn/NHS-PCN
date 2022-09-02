#http://localhost:8501

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
#nltk.download('stopwords')
conn=sqlite3.connect('data.db',check_same_thread=False,timeout=15)
curr=conn.cursor()
#op_df=table[int()].df
#st.dataframe(op_df)

model=pickle.load(open('teletubby.pkl','rb'))
#model = tf.keras.models.load_model("lstm_dataset.h5") #,custom_objects={'KerasLayer':hub.KerasLayer})
#tokenizer = Tokenizer(num_words=100)# filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#model = tf.saved_model.load(mobilenet_save_path)
st.title("Primary Care Network")

st.write("""
         Book an appointment
         """)

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



def send_mail(answer,email):
    ans = answer
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.ehlo()

    server.login('lasima.sn@gmail.com', '')

    subject = "Appointment booked!"
    body = "Your appointment has been booked with {dept} department at {time} ".format(dept=ans, time='2pm')
    msg = f"Subject: {subject}\n\n{body}"

    server.sendmail(
        'lasima.sn@gmail.com',
        email,
        msg
    )


with st.form("entry_form",clear_on_submit=True):
    #Disease,Sym1=st.columns(9)
    Name=st.text_input("Full Name")
    Disease= st.selectbox("Why are you here", ("Book a GP appointment or get health advice","Refill Prescription","Get help with MH"))
    Sym1=st.text_area("Which part of your body is affected? Explain in a few words")
    #Sym1=st.selectbox("Which part of your body is your concern associated with?", ("Wrist , hand or finger","neck or back","foot or ankle or leg","Head"))
    Sym2=st.selectbox("How long have you had your current symptoms?", ("less than 24 hrs","longer than 24 hrs","long term or intermittent"))
    Sym3=st.selectbox("Have you tried any of these medications?", ("Paracetamol","Aspirin or any other medication","Any cream or gel","Not tried"))
    #Sym4=st.selectbox("Have you been diagnosed with any long term medical problems?", ("Yes","Migrane or cluster or sinus headache","None"))
    Sym5=st.selectbox("Do you have any of the below symptoms?", ("No","Redness or Swelling or loss of sensation or painful lump","Change of shape or loss of movement"))
    Sym6=st.selectbox("Do you feel any of these difficulties accompanied with the pain?", ("No","fever lasting more than 5 days","Excessive sleepiness or change in concious level","Lack of energy","Jaw pain","Worsening headache"))
    Sym7=st.selectbox("Are you able to perform regular activities?", ("Can do regular activities","Unable to move affected area"))
    #Sym8=st.selectbox("Do you have any of these symptoms?",("None","Squint or eyes affected","Dizzy or associated with ataxia or unsteadiness","intense pain in bluish discolouration","persistent vomiting","stress","anxiety or depression"))
    Comment=st.text_area("Is there anything you want to tell us.",placeholder="Is there anything you want to tell us.")

    Email = st.text_input("Enter your email address",placeholder="Email address")
    "---"


    submitted=st.form_submit_button("Submit")
    #Save to df
    # df=pd.DataFrame(columns=['Why are you here',
    #                            'Which part of your body is affected?',
    #                            'Email',
    #                            'Department'])

    # st.download_button("Export CSV",
    #                    op_df.to_csv(),
    #                    mime='text/csv')
    if submitted:
        answer=analyse(Disease, Sym1, Sym2, Sym3, Sym5, Sym6, Sym7)
        st.write("You are being referred to {dept}. We will get back to you shortly with your appointmnet details.".format(dept=answer))
        st.success("Appointment request received")
        #code to save to sqllite
        addData(Name,Disease,Email,answer)
        send_mail(answer,Email)
        #code to convert df to csv
        # df_new = pd.DataFrame({'Why are you here': [Disease],
        #                      'Which part of your body is affected?': [Sym1],
        #                        'Email': [Email],
        #                        'Department': [answer],
        #
        #                        })
        # df=pd.DataFrame()
        # df=pd.concat([df,df_new],ignore_index=True)
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








