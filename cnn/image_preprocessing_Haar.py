import cv2
import string
import glob, os
from string import Template
from PIL import Image
from PIL import ImageFilter
import random

#Facial Recognition Model
FaceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Directories
inputDIR = "../dataset/new_photos/HAPPY_f/"
outputDIR = "../dataset/new_photos/happy_modified/"
##inputDIR = "../dataset/original_photos/angry/"
##outputDIR = "../dataset/angry/"
##inputDIR = "../dataset/original_photos/sad/"
##outputDIR = "../dataset/sad/"

#Rename files
nameTemplate_c = Template('$index.central_face.jpg') 
nameTemplate_e = Template('$index.edge_face.$loop.jpg')
nameTemplate_r = Template('$index.rotated_face.$loop.jpg') 
nameTemplate_b = Template('$index.blurred_face.$loop.jpg')

targetsize = 120, 120
left = 1
right = 1
top = 1
bottom = 1

i = 1
for infile in os.listdir(inputDIR):
    if infile.endswith(".png") or  infile.endswith(".jpg") or infile.endswith(".jpeg") or infile.endswith(".JPG"):#ifferent types of files
        inputpath = os.path.join(inputDIR, infile)
        file, ext = os.path.splitext(inputpath)
        #Load of images and transformation to greyscale
        imcv = cv2.imread(inputpath) 
        graycv = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
        heightcv, widthcv, _ = imcv.shape

        # adjust contrast, parameters of the function is tuned to give the best performance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#parameters
        graycv = clahe.apply(graycv)

        #Facial Area Detection        
        faces = FaceCascade.detectMultiScale(
            graycv, 
            scaleFactor=1.1, ##adjust to different photo size
            minNeighbors=5,
            minSize=(int(widthcv*0.2), int(heightcv*0.2)), ##minimum dimension of the faces
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if len(faces) == 1:
            for (x,y,w,h) in faces:
                # save the face with more margins for the rotation and random cutting edges
                centerw = x+w/2
                centerh = y+h/2
                left = int(centerw - (w*1.4/2))
                if left < 1: left = 1
                right = int(left + w*1.4)
                if right > widthcv: right = widthcv

                top = int(centerh - (h/2*1.4))
                if top < 1: top = 1
                bottom = int(top + h*1.4)
                if bottom > heightcv: bottom = heightcv

                if (bottom - top) != (right-left):
                    sqrdim = min((bottom - top),(right-left))
                    centerw = right/2
                    centerh = bottom/2
                    left = int(centerw - sqrdim/2)
                    right = int(centerw + sqrdim/2)
                    top = int(centerh - sqrdim/2)
                    bottom = int(centerh + sqrdim/2)
                    
                #Crop and Save
                croped = graycv[top:bottom, left:right ]
                cv2.imwrite(outputDIR+nameTemplate_c.substitute(index = i), croped)
                
                #enlarge the dataset
                impil = Image.open(outputDIR+nameTemplate_c.substitute(index = i))
                widthpil, heightpil = impil.size

                ## random rotate
                for j in range(1,5):
                        random.seed(j*10+i)
                        rotateangle = random.randrange(-45,45) 
                        out = impil.rotate(rotateangle)
                        left = int(widthpil/2-widthpil*0.7/2)
                        right = int(left + widthpil * 0.7)
                        top = int(heightpil/2 - heightpil*0.7/2)
                        bottom = int(top + heightpil*0.7)
                        box = (left, top, right, bottom)
                        out = out.crop(box)
                        out = out.resize(targetsize)
                        out.save(outputDIR + nameTemplate_r.substitute(index = i, loop = j))

                ## random cut 
                for j in range(1,4):
                        random.seed(j*10+i)
                        right = random.randrange(int(widthpil*0.7), widthpil)
                        left = int(right - widthpil*0.7)
                        bottom = random.randrange(int(heightpil*0.7), heightpil)
                        top = int(bottom - heightpil*0.7)
                        box = (left, top, right, bottom)
                        out = impil.crop(box)
                        out = out.resize(targetsize)
                        out.save(outputDIR + nameTemplate_e.substitute(index = i, loop = j))

                ## different blur
                croped = graycv[y:y+h, x:x+w ]
                cv2.imwrite(outputDIR+nameTemplate_c.substitute(index = i), croped)
                imC = Image.open(outputDIR+nameTemplate_c.substitute(index = i))
                imC = imC.resize(targetsize)
                imC.save(outputDIR+nameTemplate_c.substitute(index = i))
                for j in range(1,3):
                        random.seed(j*10+i)
                        blurradius = random.randrange(11,30)/10
                        out = imC.filter(ImageFilter.GaussianBlur(blurradius))
                        out.save(outputDIR + nameTemplate_b.substitute(index = i, loop = j))
        
            print("Picture {0}".format(i))
            i=i+1
