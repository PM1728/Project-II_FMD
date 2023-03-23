
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
import mysql.connector
from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
from time import sleep
import notify2
import subprocess
import board
import busio as io
import adafruit_mlx90614

i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)


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
	
#def openGate():
    #pwm.ChangeDutyCycle(2.0)
    #pigpio.set_PWM_dutycycle(2.0)
    #sleep(0.5)
    
    
#def closeGate():
    #pwm.ChangeDutyCycle(12.0)
    #pigpio.set_PWM_dutycycle(12.0)
    #sleep(0.1)


#def closeEverything():
		#GPIO.output(redLed, GPIO.LOW)
		#GPIO.output(greenLed, GPIO.LOW)
		#GPIO.output(buzz, GPIO.LOW)
		#closeGate()
        


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
			targetTemp = "{:.2f}".format(mlx.object_temperature)

			sleep(1)
			print("Target Temperature:", targetTemp,"°C")
			

		
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
			#dist = GPIO.input(ir)

			#if dist == 0:
				#applyLogic(label)
			#else:
				#closeEverything()
        
   	
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
					
				
	# กดแล้ว up to database
	# ข้อความบน program 3 กรณี มุมขวาบน

   
	keyCap = cv2.waitKey(1) & 0xFF
	if keyCap == ord("c"):		
	# Save the image to a file
		file_name = "captured_image_" + str(int(time.time())) + ".jpg"
		file_path = "face mask detection" + file_name
		cv2.imwrite(file_path, image)
	print("Detected and image saved")
 


 			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
            break

vs.release()
cv2.destroyAllWindows()
#pwm.stop()
GPIO.cleanup()
vs.stop()
#Testing