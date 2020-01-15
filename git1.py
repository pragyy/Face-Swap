import cv2
from faceSwap import swap

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt(2).xml")

while(True):
	# Capture frame-by-frame
    ret, frame = cap.read()

	# Our operations on the frame come here
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

    print("Found {0} faces!".format(len(faces)))
    #1.crop the image
    #2.pass it through the classifier   clf.predict(cropped)
    #4.
	# Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (54, 87, 999), 2)
    #print(faces)
    if len(faces)==2:
        x,y,w,h=faces[0]
        croppedimg=frame[y:y+h,x:x+w]
        cv2.imwrite("1.jpg",croppedimg)
        x,y,w,h=faces[1]
        croppedimg2=frame[y:y+h,x:x+w]
        cv2.imwrite("2.jpg",croppedimg2)
        image1=cv2.imread("1.jpg")
        image2=cv2.imread("2.jpg")
        x_offset,y_offset=x,y
        print(x_offset)
        print(y_offset)
        frame=frame[y_offset:y_offset+image1.shape[0],x_offset:x_offset+image1.shape[1]]
        output = swap("1.jpg","2.jpg")
        cv2.imwrite('output.jpg',output)
	# Display the resulting frame
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()