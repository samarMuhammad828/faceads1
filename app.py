from flask import Flask, render_template, jsonify, request
from PIL import Image
import base64
import io
# facerec.py
#import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable

#from PIL import Image
#import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#detector = MTCNN()
#dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')

app = Flask(__name__)



@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

test_transforms = transforms.Compose([
        #transforms.Resize((128,128)),
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
print('kkkkkkkkkk')    



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
    
#background process happening without any refreshing

@app.route('/snap_a_signal', methods=["POST", "GET"])
def process_signal():
    pixels = request.get_json()['data']
    result=im2info(pixels)
    if result:
        print(result)

        return jsonify(gender=result[0],
                       glass=result[1],
                       chubby=result[2]
                       )
    #else:
    #    return jsonify(result = "0");
    print('kkkkkkkkkk2')    

def im2info(pixels ):
    #try:
    #image_data = dataUrlPattern.match(pixels).group(2)
    #image_data = image_data.encode()
    image_data = base64.b64decode(pixels.split(",")[1])
    #print(type(image_data))
    #with open('screenshot.jpg', 'wb') as f:
    #    f.write(image_data)
#        
#        
    print('kkkkkkkkkk300000')        
    #fh = open("imageToSave.png", "wb")
    #fh.write(str(pixels.split(",")[1].decode('base64')))
    #fh.close()
    #img = cv2.imread("screenshot.jpg")
    #image = Image.fromstring('F', (300,300), pixels, 'raw', 'F;16')
    #image.convert('L').save('out.png')
   
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    
#        faces = face_cascade.detectMultiScale(
#                gray,
#                scaleFactor=1.1,
#                minNeighbors=5,
#                minSize=(15, 15)
#            )
    image = Image.open(io.BytesIO(image_data))
    #print(image.shape)
    #image = Image.open('imageToSave.png')
    background = image.convert('RGB')

    #png = Image.open(object.logo.path)
    #image.load() # required for png.split()
    
    #background = Image.new("RGB", image.size, (255, 255, 255))
    #background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
    #a = numpy.array(background)# a is readonly
    print('kkkkkkkkkk3')    


    #del pixels, image_data, image
    #faces = detector.detect_faces(a)

    #if len(faces) > 0:
       # x,y,w,h = faces[0]['box']

        #x,y,w,h = faces[0]
        #if w > 20 :
        #    imsh = a.shape
      #newimg = img[max(0,(y-20)):min(imsh[0],(y+h+20)),
                   #max(0,(x-20)):min(imsh[1],(x+w+20)),]
         #   area = (max(0,(x-20)),
          #          max(0,(y-20)),
           #         min(imsh[1],(x+w+20)),
            #        min(imsh[0],(y+h+20)))
            #PIL_image = background[y:y+h,x:x+w]
            #cropped_img = background.crop(area)
    background.save('out.png')
            
            #cropped_img.show()
            #PIL_image = Image.fromarray(newimg)

    result = [predict_gender(background),
              predict_glass(background),
              predict_Chubby(background)]
    
    
    return result
    #except:
        #result = [".",".","."]
       # return result
    
    
    
    
if __name__ == '__main__':
    app.run()
    app.debug = True