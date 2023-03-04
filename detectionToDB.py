import cv2
import dlib

# Load the pre-trained models for face detection and mask detection
face_detector = cv2.CascadeClassifier('face mask recognition\haarcascade_frontalface_default.xml')
mask_detector = cv2.CascadeClassifier('face mask recognition\haarcascade_frontalface_default.xml')

# Load the pre-trained model for face recognition
face_recognizer = dlib.face_recognition_model_v1('face mask recognition/dlib_face_recognition_resnet_model_v1.dat')
sp = dlib.shape_predictor('face mask recognition/shape_predictor_68_face_landmarks.dat')

# Database of known faces
known_faces = {
    'name1': [face_descriptor1, face_descriptor2, ...],
    'name2': [face_descriptor3, face_descriptor4, ...],
    
}

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Iterate over the faces and check if they are wearing masks
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect masks in the region of interest
        masks = mask_detector.detectMultiScale(roi_gray)

        # Draw a rectangle around the face if no mask is detected
        if len(masks) == 0:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Recognize the face
        shape = sp(frame, dlib.rectangle(x, y, x+w, y+h))
        face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)

        # Compare the face descriptor with the database of known faces
        best_match = None
        best_distance = 1.0
        for name, face_descriptors in known_faces.items():
            for known_face_descriptor in face_descriptors:
                distance = dlib.vector_distance(face_descriptor, known_face_descriptor)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name

        # Draw the name of the recognized person on the frame
       
        # Draw the name of the recognized person on the frame
        if best_match is not None:
            cv2.putText(frame)

