import cv2
from os.path import dirname, join


def preprocess_face_detector(frame):
    mean_face_detector = (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), mean_face_detector)
    return blob


def preprocess_age_gender_classifier(frame):
    mean_age_classifier = (78.4263377603, 87.7689143744, 114.895847746)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), mean_age_classifier, swapRB=False)
    return blob


gender_classifier = cv2.dnn.readNet("gender_deploy.prototxt", "gender_net.caffemodel")
age_classifier = cv2.dnn.readNet("age_deploy.prototxt", "age_net.caffemodel")
protoPath = join(dirname(__file__), "opencv_face_detector.pbtxt")
modelPath = join(dirname(__file__), "opencv_face_detector_uint8.pb")
face_detector = cv2.dnn.readNetFromTensorflow(modelPath, protoPath)


confidence = 0.65
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_detector.setInput(preprocess_face_detector(frame))
        detections = face_detector.forward()
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > confidence:
                (h, w) = frame.shape[:2]
                x = int(detections[0, 0, i, 3]*w)
                y = int(detections[0, 0, i, 4]*h)
                X = int(detections[0, 0, i, 5]*w)
                Y = int(detections[0, 0, i, 6]*h)
                face = frame[y:Y, x:X]
                preprocessed_face = preprocess_age_gender_classifier(frame)
                age_classifier.setInput(preprocessed_face)
                age_prob = age_classifier.forward()
                age = ageList[age_prob[0].argmax()]
                gender_classifier.setInput(preprocessed_face)
                gender_prob = gender_classifier.forward()
                gender = genderList[gender_prob[0].argmax()]
                cv2.rectangle(frame, (x, y), (X, Y), (0, 255, 0), thickness=2)
                y_text = y - 10 if y - 10 > 10 else y + 10
                cv2.putText(frame, "age: {}; gender: {}".format(age, gender), (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0, 255, 0), 2)
        cv2.imshow("Face-Detection-Age-Gender-Classification", frame)
        if cv2.waitKey(25) & 0xFF == ord(' '):
            break
    else:
        cv2.imshow(frame)

cap.release()


