import cv2
import os
from flask import Flask, request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# defining flask app
app = Flask(__name__)

#saving date in two format
def datetoday():
    return date.today().strftime("%m_%d_%y")
def datetoday2():
    return date.today().strftime("%d-%B-%Y")

#initializing videocapture object to access webcam
face_detector =cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

#directories don't exist
if not os.path.isdir('Project/Attendance'):
    os.makedirs('Project/Attendance')
if not os.path.isdir('Project/static/faces'):
    os.makedirs('Project/static/faces')

if f'Attendance-{datetoday()}.csv' not in os.listdir('Project/Attendance'):
    with open(f'Project/Attendance/Attendance-{datetoday()}.csv','w') as f:
        f.write('Name,Roll,Time')

#get a number of total registered user
def totalreg():
    return len(os.listdir('Project/static/faces'))

#extracting the face from image
def extract_faces(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

#identifying face using Ml model
def identify_face(facearray):
    model = joblib.load('Project/static/face_recognition_model.pkl')
    return model.predict(facearray)

#function which trains model on all available faces
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('Project/static/faces')
    for user in userlist:
        for imgname in os.listdir(f'Project/static/faces/{user}'):
            img = cv2.imread(f'Project/static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'Project/static/face_recognition_model.pkl')

# extract info from todays attendance file in attendance folder 
def extract_attendance():
    df = pd.read_csv(f'Project/Attendance/Attendance-{datetoday()}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Project/Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Project/Attendance/Attendance-{datetoday()}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


#   *****routing function*****

#Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2())

# This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('Project/static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()

        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            if identified_person == 'unknown':
                cv2.putText(frame, 'Unauthorized', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                # Perform some action for unknown person
                # For example, you can log the event or trigger an alarm
                # Add your code here
            else:
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)

        cv2.imshow('Project/Attendance', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2())


# This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'Project/static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)