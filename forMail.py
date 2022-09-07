import re
import smtplib



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
        'nhs.fortest@gmail.com',
        email,
        msg
    )
