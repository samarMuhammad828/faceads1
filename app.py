from flask import Flask, render_template, jsonify, request
from PIL import Image
import re
from face_recognize import predict_gender, predict_glass, predict_Chubby
import base64
import io
import numpy
from mtcnn.mtcnn import MTCNN
app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

#background process happening without any refreshing

@app.route('/snap_a_signal', methods=["POST", "GET"])
def process_signal():
    pixels = request.get_json()['data']
    detector = MTCNN()
    result=im2info(pixels, detector)
    
    if result:
        return jsonify(gender=result[0],
                       glass=result[1],
                       chubby=result[2]
                       )
    else:
        return jsonify(result = "0");

def im2info(pixels, detector):
    #try:
    dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
    image_data = dataUrlPattern.match(pixels).group(2)
    image_data = image_data.encode()
    image_data = base64.b64decode(image_data)
#        
#    with open('screenshot.jpg', 'wb') as f:
#        f.write(image_data)
##        
#        
        
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
    #image = Image.open('imageToSave.png')
    
    #png = Image.open(object.logo.path)
    image.load() # required for png.split()

    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
    a = numpy.array(background)# a is readonly
    
    faces = detector.detect_faces(a)

    if len(faces) > 0:
        x,y,w,h = faces[0]['box']

        #x,y,w,h = faces[0]
        if w > 10 :

            newimg = a[y:y+h,x:x+w]
            print(newimg.shape)
            PIL_image = Image.fromarray(newimg)

            result = [predict_gender(PIL_image),
                      predict_glass(PIL_image),
                      predict_Chubby(PIL_image)]
    
    
            return result
    #except:
        #result = [".",".","."]
       # return result
    
    
    
    
if __name__ == '__main__':
    app.run()
    app.debug = True