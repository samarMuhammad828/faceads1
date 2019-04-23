# facerec.py
#import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

# Model class must be defined somewhere
model_gender = torch.load('model_gender_ft.pkl', map_location='cpu')
model_gender.eval()


model_glass = torch.load('modelglass_ft.pkl', map_location='cpu')
model_glass.eval()


model_Chubby = torch.load('modelChubby_ft.pkl', map_location='cpu')
model_Chubby.eval()


def predict_gender(image):
    image_tensor = test_transforms(image).float()
    #print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze_(0)
    #print(image_tensor.shape)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model_gender(input)
    _, preds = torch.max(output, 1)
    #print(_)
    #print(preds)
    index = output.data.cpu().numpy()
    if index.argmax() == 0:
        return "woman"
    else:
        return "man"
    
    
def predict_glass(image):
    image_tensor = test_transforms(image).float()
    #print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze_(0)
    #print(image_tensor.shape)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model_glass(input)
    _, preds = torch.max(output, 1)
    #print(_)
    #print(preds)
    index = output.data.cpu().numpy()
    if index.argmax() == 0:
        return "wearing glasses"
    else:
        return "no glasses"
    
def predict_Chubby(image):
    image_tensor = test_transforms(image).float()
    #print(image_tensor.shape)
    image_tensor = image_tensor.unsqueeze_(0)
    #print(image_tensor.shape)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model_Chubby(input)
    _, preds = torch.max(output, 1)
    #print(_)
    #print(preds)
    index = output.data.cpu().numpy()
    if index.argmax() == 0:
        return "Chubby"
    else:
        return "no Chubby"
    
#size = 4
#haar_file = 'haarcascade_frontalface_default.xml'
#datasets = 'datasets'
## Part 1: Create fisherRecognizer
#print('Recognizing Face Please Be in sufficient Light Conditions...')
## Create a list of images and a list of corresponding names
#(images, lables, names, id) = ([], [], {}, 0)
#for (subdirs, dirs, files) in os.walk(datasets):
#    for subdir in dirs:
#        names[id] = subdir
#        subjectpath = os.path.join(datasets, subdir)
#        for filename in os.listdir(subjectpath):
#            path = subjectpath + '/' + filename
#            lable = id
#            images.append(cv2.imread(path, 0))
#            lables.append(int(lable))
#        id += 1
#(width, height) = (130, 100)
#
## Create a Numpy array from the two lists above
#(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
#model = cv2.face.LBPHFaceRecognizer_create()
#model.train(images, lables)


t1 = time.time()
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Part 2: Use fisherRecognizer on camera stream
#face_cascade = cv2.CascadeClassifier(haar_file)
#webcam = cv2.VideoCapture(0)
#while True:
#    (_, im) = webcam.read()
#    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(
#        gray,
#        scaleFactor=1.1,
#        minNeighbors=5,
#        minSize=(30, 30))
#    if len(faces) > 0:
#        x,y,w,h = faces[0]
#        t2 = time.time()
#        if w > 20 and t2 - t1 > 1:
#            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#            newimg = im[y:y+h,x:x+w]
#            PIL_image = Image.fromarray(newimg)
#            p1 = predict_gender(PIL_image)
#            p2 = predict_glass(PIL_image)
#            p3 = predict_Chubby(PIL_image)
#            t1 = time.time()
#
#    for (x, y, w, h) in faces:
#        cv2.putText(im,p1,(x-10, y-30), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
#        cv2.putText(im,p2,(x-10, y-20), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
#        cv2.putText(im,p3,(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
#
#        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#    # Display the resulting frame
#    cv2.imshow('Video', im)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#
## When everything is done, release the capture
#webcam.release()
#cv2.destroyAllWindows()