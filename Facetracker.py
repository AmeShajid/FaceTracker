# Import the OpenCV library
import cv2

# Load the pre-trained face cascade classifier XML file
face_cascade = cv2.CascadeClassifier('/Users/ameshajid/Documents/VisualStudioCode/Small Projects/Trackers/haarcascade_frontalface_default.xml')

# Initialize the webcam. '0' indicates the default camera.
webcam = cv2.VideoCapture(0)

# Start an infinite loop to continuously capture frames from the webcam
while True:
    # Capture frame-by-frame from the webcam
    ret, frame = webcam.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the captured frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the face cascade classifier
    # Parameters:
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # minSize: Minimum possible object size. Objects smaller than this size will be ignored.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and add text
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text 'Face' above the face rectangle
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display the resulting frame with the detected faces
    cv2.imshow("Face detection", frame)

    # Wait for a key press. If 'Esc' key (27) is pressed, exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()


