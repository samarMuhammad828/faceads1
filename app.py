from flask import Flask, render_template, jsonify, request
import cv2
from PIL import Image
import re
from face_recognize import face_cascade, predict_gender, predict_glass, predict_Chubby
import base64


app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

#background process happening without any refreshing

@app.route('/snap_a_signal', methods=["POST", "GET"])
def process_signal():
    pixels = request.get_json()['data']
    result=im2info(pixels)
    if result:
        return jsonify(gender=result[0],
                       glass=result[1],
                       chubby=result[2]
                       )
    else:
        return jsonify(result = "0");

def im2info(pixels):
    dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
    image_data = dataUrlPattern.match(pixels).group(2)
    image_data = image_data.encode()
    image_data = base64.b64decode(image_data)
 
    with open('screenshot.jpg', 'wb') as f:
        f.write(image_data)

    img = cv2.imread("screenshot.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    if len(faces) > 0:
        

        x,y,w,h = faces[0]
        if w > 20 :

            newimg = img[y:y+h,x:x+w]
            PIL_image = Image.fromarray(newimg)

            result = [predict_gender(PIL_image),
                      predict_glass(PIL_image),
                      predict_Chubby(PIL_image)]
    
    
            return result
    
    
    
    
if __name__ == '__main__':
    app.run()
    app.debug = True