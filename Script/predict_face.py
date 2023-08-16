import cv2 
import os
import numpy as np
import mtcnn
from architecture import *
from scipy.spatial.distance import cosine
#from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
import pickle
#import time
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

confidence_t=0.9
recognition_t=0.5
required_size = (160,160)
l2_normalizer = Normalizer('l2')

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

def detect(img ,detector,encoder,encoding_dict, filename):
    match = False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'Inconnu'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'Inconnu':
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 200), 1)
            new_filename = os.path.splitext(filename)[0] + "_non.jpg"
        else:
            match = True
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name + f'_{1-distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            new_filename = os.path.splitext(filename)[0] + "_oui.jpg"
    return img, new_filename, match 

def add_face_embedding(image_path):
    face_name = os.path.basename(image_path)
    face_name = os.path.splitext(face_name)[0]
    img_BGR = cv2.imread(image_path)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

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

if __name__ == "__main__":
    
    #start = time.time()
    
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

    # read the file
    temp_path = os.path.join(working_dir, 'temp_img')
    for file in os.listdir(temp_path):
        if file.endswith(".jpg"):
            filename = os.path.join(temp_path, file)
            frame = cv2.imread(filename)
            basename = os.path.basename(filename)
            frame, new_filename, match = detect(frame , face_detector , face_encoder , encoding_dict, basename)

            #imgplot = plt.imshow(frame)
            #plt.show()

            new_path = os.path.join(working_dir, "temp_img")
            new_path = os.path.join(new_path, new_filename)
            cv2.imwrite(new_path, frame)
            
            # add the face on the dict
            print(new_filename)
            if match == False: 
                add_face_embedding(filename)
    
    # save the encodings    
    with open(encodings_path, 'wb') as file:
        pickle.dump(encoding_dict, file)
    
    #end = time.time_ns()
    #print(f"Frame Time : {end-start}")
    