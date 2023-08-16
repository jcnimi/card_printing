import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from architecture import *
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import time

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

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

start = time.time()

working_dir = os.getcwd()
required_shape = (160,160)
print("Loading facenet on path ", os.path.join(working_dir, "facenet_model.pkl"))
# face_encoder = load_pickle(os.path.join(working_dir, "facenet_model.pkl"))

face_encoder = InceptionResNetV2()
path_m =  os.path.join(working_dir,"facenet_keras.h5")
face_encoder.load_weights(path_m)

face_detector = mtcnn.MTCNN()
encodes = []
l2_normalizer = Normalizer('l2')

encodings_path = 'encodings.pkl'
print("encoding path:",os.path.join(working_dir,'encodings.pkl'))
if os.path.exists(os.path.join(working_dir,'encodings.pkl')):
    print("Encoding exist")
    encoding_dict = load_pickle(encodings_path)
else:
    print("Dictionaire des visage non trouvé, creation")
    encoding_dict = dict()

# Get the file
temp_path = os.path.join(working_dir, 'temp_img')
print("Entrainement du modele en cours...")
for file in os.listdir(temp_path):
    if file.endswith(".jpg"):
        filename = os.path.join(temp_path, file)
        print("Ajout de {0}".format(filename))
        add_face_embedding(filename)

with open(encodings_path, 'wb') as file:
    pickle.dump(encoding_dict, file)
    
end = time.time()

#print("Running time: ", end - start)




