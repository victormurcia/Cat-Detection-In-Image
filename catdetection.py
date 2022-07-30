import cv2

path2img = r'C:\Users\vmurc\Pictures\Catterina\inbound7822203518714780106.jpg'
path2xml = r'C:\Users\vmurc\Documents\GitHub\opencv\data\haarcascades\haarcascade_frontalcatface_extended.xml'

catFaceCascade = cv2.CascadeClassifier(path2xml)
# load the input image and convert it to grayscale
image = cv2.imread(path2img)
n=1
dim = (n*813,n*1129)#(width, height)  
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
sf = 1.5
for i in range(50):
    if sf <= 1:
        break
    faces = catFaceCascade.detectMultiScale(resized, scaleFactor = sf, minSize=(200, 200), minNeighbors=0)
    if len(faces) == 0:
        print("No faces found", 'SF = ' + str(sf))
        sf -= 0.01
    elif len(faces) > 1:
        print("More than 1 face found. Try increasing initial value of the scale factor.", 'SF = ' + str(sf))
        sf += 0.001
    else:
        print("Number of faces detected: " + str(faces.shape[0]))        
        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0),4)

        cv2.imshow('Image with faces', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
        break
