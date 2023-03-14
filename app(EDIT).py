
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
from datetime import datetime
from datetime import date
import face_recognition
import mysql.connector

FONT=cv2.FONT_HERSHEY_COMPLEX
#number=[]
#status=[]
images=[]
names=[]
today = date.today()
now = datetime.now()
dtString = now.strftime("%H:%M:")
path = "\images*.*"
for file in glob.glob(path):
    image = cv2.imread(file)
    a=os.path.basename(file)
    b=os.path.splitext(a)[0]
    names.append(b)
    images.append(image)
def encoding1(images):
    encode=[]

    for img in images:
        unk_encoding = face_recognition.face_encodings(img)[0]
        encode.append(unk_encoding)
    return encode    

encodelist=encoding1(images)
def mysqladddata(names):
       mydb = mysql.connector.connect(
       host= "127.0.0.1",
       user="root",
       password="",
       database="data_db"

)

       a = mydb.cursor()
       sql = ("INSERT IGNORE INTO users(Date,Time) VALUE(%s,%s)")
       data=(today,dtString)

       a.execute(sql,data)
       

       mydb.commit()
       mydb.close()

nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')


def detect_and_predict_mask(frame, faceNet, maskNet):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	
	faces = []
	locs = []
	preds = []


	for i in range(0, detections.shape[2]):
     
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
	
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))


	if len(faces) > 0:
		
	
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)


	return (locs, preds)


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


print("[INFO] ...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

Msk =8
Mnt =8
Nos =8

# loop over the frames video stream
while True:
	
	faces = []
	image = vs.read()
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	frame1=cv2.resize(frame,(0,0),None,0.25,0.25)
	face_locations = face_recognition.face_locations(frame1)
	curframe_encoding = face_recognition.face_encodings(frame1,face_locations)
	for encodeface,faces in zip(curframe_encoding,face_locations):
         results = face_recognition.compare_faces(encodelist, encodeface)
         distance= face_recognition.face_distance(encodelist, encodeface)
         mysqladddata(0)
         
        
        
         
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)


	for (box, pred) in zip(locs, preds):
		
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

		
			label = "Mask" if mask > withoutMask else "No Mask" 
			if (label == 'Mask' ):
				Msk = 1
			else: Msk =0 
			print('Mask =',Msk)

			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   	
		#nose
	for (box2, detect_nose) in zip(locs, preds):
		
			(startX, startY, endX, endY) = box2
			(no_nose_rects, nose_rects) = detect_nose
   
			(no_nose_rects, nose_rects) = detect_nose
   
			N = "Nose" if nose_rects > no_nose_rects else "No nose"	
			if (N == 'Nose' ):
				Nos = 1
			else: Nos =0 
			print("Nose =",Nos)
			nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in nose_rects:
					cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
					break
		
  		#mouth
	for (box3, detect_mouth) in zip(locs, preds):
		
			(startX, startY, endX, endY) = box3
			(no_mouth_rectes, mouth_rectes) = detect_mouth
 
			M = "Mouth" if mouth_rectes > no_mouth_rectes else "No Mouth"
			if (M == 'Mouth' ):
				Mnt = 1
			else: Mnt =0 
			print("Mouth =",Mnt)
	  
			
			mouth_rectes = mouth_cascade.detectMultiScale(gray, 1.1, 20)
			for (x,y,w,h) in mouth_rectes:
					cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
					break
				
	# กดแล้ว up to database
	# ข้อความบน program 3 กรณี มุมขวาบน

   
	
		
    # Save the image to a file
	file_name = "captured_image_" + str(int(time.time())) + ".jpg"
	#file_path = "face mask recognition" + file_name
	#cv2.imwrite(file_path, image)
	print("Detected and image saved")


 			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break	

vs.release()
cv2.destroyAllWindows()
vs.stop()
#Testing