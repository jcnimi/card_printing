from flask import Flask, request
import cv2 
import os
import numpy as np
import mtcnn
from architecture import *
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
import pickle
import io
import base64 
from PIL import Image
import logging


confidence_t=0.9
recognition_t=0.8
required_size = (160,160)
l2_normalizer = Normalizer('l2')

app = Flask(__name__)

auth_ref = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImpjbmltaSIsImlhdCI6MTY2NjAwNDc5OCwiZXhwIjoxNjY2MDExOTk4fQ.WPzzenZp8PA28E-kEaqND4fikYJkJzEdaGEwR88lSys"

#logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s, %(message)s')

logging.basicConfig(level=logging.DEBUG)
log_format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
formatter = logging.Formatter(log_format)

logger = logging.getLogger('werkzeug') # grabs underlying WSGI logger
handler = logging.FileHandler('app.log') # creates handler for the log file
handler.setFormatter(formatter)
logger.addHandler(handler) # adds handler to the werkzeug WSGI logger

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(imgdata)).convert('RGB')
    img = np.array(img) 
    return img

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    #return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector,encoder,encoding_dict, filename):
    match = False
    img_rgb = img
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'Inconnu'

        distance = float("inf")
        app.logger.info("name before")
        app.logger.info(encoding_dict.items())
        app.logger.info(name)
        for db_name, db_encode in encoding_dict.items():
            app.logger.info(db_name)
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
        img = toRGB(img)
        app.logger.info("name after")
        app.logger.info( encoding_dict.items())
        app.logger.info(name)
        
        if name == 'Inconnu':
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 200), 1)
        else:
            match = True
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name + f'_{1-distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
    return img, match 

def add_face_embedding(img, face_name):
    img_RGB = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    x = face_detector.detect_faces(img_RGB)
    x1, y1, width, height = x[0]['box']
    x1, y1 = abs(x1) , abs(y1)
    x2, y2 = x1+width , y1+height
    face = img_RGB[y1:y2 , x1:x2]
            
    face = normalize(face)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    encode = face_encoder.predict(face_d)[0]
    encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_name] = encode
        

#POST REQUEST
@app.route('/predict', methods = ['POST'])
def postRequestPredict():
    app.logger.info(request.headers)
    content_type = request.headers.get('Content-Type')
    
    try:
        if (content_type == 'application/json'):
            json = request.json
            src_img_base64 = json["img"]
            name = json["name"]
            edit_mode = json["edit"]
            
            srcImg = stringToImage(src_img_base64)
            #app.logger.info(src_img_base64)
            outImg, match = detect(srcImg , face_detector , face_encoder , encoding_dict, name)        

            if match == False and edit_mode == 1: 
                add_face_embedding(srcImg, name)
        
            # save the encodings    
            with open(encodings_path, 'wb') as file:
                pickle.dump(encoding_dict, file)
            
            #im_pil = Image.fromarray(outImg)
            # im_pil.save("pil_img.jpg")

            im_pil2 = Image.fromarray(toRGB(outImg))
            # im_pil2.save("pil_img2.jpg")

            buffered = io.BytesIO()
            im_pil2.save(buffered, format="JPEG")
            out_img_base64 = base64.b64encode(buffered.getvalue())

            if match:
                match_val = 1
            else:
                match_val = 0

            return {"img": out_img_base64.decode('utf-8'), "match": match_val}
        else:
            return 'Content-Type not supported!'
    except Exception as ex:
        app.logger.error(str(ex))

if __name__ == "__main__":
    
    working_dir = os.getcwd()
    
    face_encoder = InceptionResNetV2()
    path_m =  os.path.join(working_dir,"facenet_keras.h5")
    face_encoder.load_weights(path_m)
    required_shape = (160,160)
    face_detector = mtcnn.MTCNN()  
    
    encodings_path =  os.path.join(working_dir,'encodings.pkl')
    if os.path.exists(encodings_path):
        encoding_dict = load_pickle(encodings_path)
    else:
        encoding_dict = dict()
    
    encodes = []
    l2_normalizer = Normalizer('l2')
    
    app.debug = True
    app.run(host='0.0.0.0', port=5000)