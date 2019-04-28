from flask import Flask, render_template, jsonify, request
from PIL import Image
import base64
import io
# facerec.py
#import cv2

from face_recognize import eval_models, \
predict_gender, predict_glass, predict_chubby, predict_look, predict_Receding_Hairline, \
predict_Bags_Under_Eyes, predict_Bald, predict_Young, predict_Pale_Skin

eval_models()

app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)


@app.route('/snap_a_signal', methods=["POST", "GET"])
def process_signal():
    pixels = request.get_json()['data']
    selected_model = request.get_json()['selected_model']
    results, accuracy=im2info(pixels, selected_model)
    print(results)
    print(accuracy)
    if results:
        print(results)

        return jsonify(results=results,
                       accuracy=accuracy
                       )

    print('kkkkkkkkkk2')    

def im2info(pixels, selected_model ):

    image_data = base64.b64decode(pixels.split(",")[1])
      
    print('kkkkkkkkkk300000')        

    image = Image.open(io.BytesIO(image_data))

    background = image.convert('RGB')

    print('kkkkkkkkkk3')    


    background.save('out.png')
            
            #cropped_img.show()
            #PIL_image = Image.fromarray(newimg)
    if selected_model == 'gender':
        return predict_gender(background)
    
    elif selected_model == 'look':
        return predict_look(background)
    
    elif selected_model == 'chubby':
        return predict_chubby(background)
    elif selected_model == 'glass':
        return predict_glass(background)
    
    
    elif selected_model == 'Receding_Hairline': 
        return predict_Receding_Hairline(background)
    
    elif selected_model == 'Bags_Under_Eyes': 
        return predict_Bags_Under_Eyes(background)
    
    elif selected_model == 'Bald': 
        return predict_Bald(background)
    
    elif selected_model == 'Young': 
        return predict_Young(background)
    
    elif selected_model == 'Pale_Skin': 
        return predict_Pale_Skin(background)




    #except:
        #result = [".",".","."]
       # return result
    
    
    
    
if __name__ == '__main__':
    app.run()
    app.debug = True