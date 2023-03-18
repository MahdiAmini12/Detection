import cv2

# load trained cascade classifier to recognize faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# start capturing video from default camera (usually the built-in webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # capture each frame of video
    ret, frame = video_capture.read()

    # detect faces in the current video frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # draw rectangles around detected faces
    for (x, y, w, h) in faces:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display the updated video frame
    cv2.imshow('Face Detection', frame)

    # break out of the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
