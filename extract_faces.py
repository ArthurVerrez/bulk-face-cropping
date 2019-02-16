
#Path to the primary folder
path="C:/Users/Arthur Verrez/Drive/Projets/Programmation/Python/FaceRecognition/ExtractFaces/"

#Path to the folder containing the images where the face needs to be cropped out of
to_crop_path=path+"faces_to_crop/"

#Path to the folder containing the cropped images
save_path=path+"saved_faces/"

import face_recognition
import imageio
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import scipy.misc
from os import listdir
from os.path import isfile, join


#Crop the image im with the variables contained in loc
# loc  (top, right, bottom, left) 
def crop_image(im, loc):
    (top, right, bottom, left) = loc
    im_crop=np.zeros((bottom-top, right-left, 3))
    for i in range(top, bottom):
        for j in range(left, right):
            im_crop[i-top,j-right]=im[i,j]
    return im_crop.astype(np.uint8)



files_to_crop = [f for f in listdir(to_crop_path) if isfile(join(to_crop_path, f))]

#Image to do the face recognition of the person whose face is to be cropped
known_image = face_recognition.load_image_file(path+"lolo.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]


#Helps with the names of the cropped images
max_k=0


#f will go over the image files to crop
for f in files_to_crop:
    image = face_recognition.load_image_file(to_crop_path+f)
    
    #Recognize the encodings of the faces in image
    encodings = face_recognition.face_encodings(image)
    
    
    im = imageio.imread(to_crop_path+f)
    
    #Take the face locations in the image, those are in the same order as the encodings
    face_locations = face_recognition.face_locations(image) # (top, right, bottom, left)
    
    #We compare each face in the image with the known encoding
    results = face_recognition.compare_faces(encodings, known_encoding)
    
    n=len(face_locations)
    im_crop=[]
    
    #We only crop the faces that correspond to the known encoding
    for i in range(len(results)):
        if(results[i]):
            im_crop.append(crop_image(im, face_locations[i]))
    
    #The cropped images are saved with a unique name
    for k in range(len(im_crop)):
        scipy.misc.imsave(save_path+str(k+max_k)+".jpg", im_crop[k])
    max_k+=len(im_crop)

