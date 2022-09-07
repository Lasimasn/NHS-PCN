from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import google_auth_oauthlib
from datetime import datetime, timedelta

# scope=['https://www.googleapis.com/auth/calendar']
# #flow=InstalledAppFlow.from_client_secrets_file("client_secret.json",scopes=scope)
# flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file("client_secret.json",scopes=scope)
#
# credentials= flow.run_local_server()
# #credentials=pickle.load(open("calendartoken.pkl","rb"))
# #Creating a sevrice object
# service =build("calendar","v3",credentials=credentials)
# now =datetime.utcnow().isoformat()+'Z'
# events_result = service.events().list(calendarId='primary', timeMin=now,
#                                               maxResults=10, singleEvents=True,
#                                               orderBy='startTime').execute()
# events = events_result.get('items', [])
#
# st=datetime(2022,9,5,9,0,0)
# et=datetime(2022,9,5,20,0,0)
# duration=timedelta(hours=1)

def parseDate(rawDate):
    return datetime.strptime(rawDate[:-6]+ rawDate[-6:].replace(":",""), "%Y-%m-%dT%H:%M:%S%z")

def get_calendar_events():
    # code to get current calendar events ans start and end date
    scope = ['https://www.googleapis.com/auth/calendar']
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", scopes=scope)
    # flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file("client_secret.json",scopes=scope)

    credentials = flow.run_local_server()
    # credentials=pickle.load(open("calendartoken.pkl","rb"))

    # Creating a sevrice object
    service = build("calendar", "v3", credentials=credentials)
    now = datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(calendarId='primary',
                                          maxResults=10, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    start = datetime(2022, 9, 6, 9, 0, 0)
    end = datetime(2022, 9, 6, 20, 0, 0)
    duration = timedelta(minutes=30)
    return service,events,start,end, duration


def findFirstOpenSlot(events, work_startTime, work_endTime, duration):
    eventStarts = [parseDate(e['start'].get('dateTime', e['start'].get('date'))) for e in events]
    eventEnds = [parseDate(e['end'].get('dateTime', e['end'].get('date'))) for e in events]
    gaps = [start - end for (start, end) in zip(eventStarts[1:], eventEnds[:-1])]

    eventStartTime = eventStarts[0].replace(tzinfo=None)
    eventEndTime = eventEnds[-1].replace(tzinfo=None)

    # print("ettt", eventEndTime + duration)
    # if gaps==[]:
    #     return work_startTime

    # if eventEndTime + duration <= work_endTime:
    #     print("end", eventEndTime)
    #     return eventEndTime

    if work_startTime + duration < eventStartTime:
        # A slot is open at the start of the desired window.
        print("st00", work_startTime)
        return startTime
    for i, gap in enumerate(gaps):
        if gap >= duration:
            # This means that a gap is bigger than the desired slot duration, and we can "squeeze" a meeting.
            # Just after that meeting ends.
            print("ev", eventEnds[i])
            return eventEnds[i]

    else:
        print("Date +1")
        work_startTime = work_startTime + timedelta(days=1)
        work_endTime = work_endTime + timedelta(days=1)
        return None
    #If no suitable gaps are found, return none.


def insert_new_calendar_event(service,Name,Sym1,Email,events, st, et, duration):
    start_time = findFirstOpenSlot(events=events, work_startTime=st, work_endTime=et, duration=duration)
    end_time = start_time + timedelta(minutes=30)
    timezone = 'Europe/London'

    events = {
        'summary': Name,
        'location': Email,
        'description': Sym1,
        'start': {
            'dateTime': start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'timeZone': timezone,
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }
    service.events().insert(calendarId='primary', body=events).execute()
    return start_time.strftime("%Y-%m-%dT%H:%M:%S"),end_time.strftime("%Y-%m-%dT%H:%M:%S")
