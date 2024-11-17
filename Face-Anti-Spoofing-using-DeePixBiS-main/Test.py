import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss

# Import the required modules
from IPython.display import clear_output
import socket
import sys
import matplotlib.pyplot as plt
import pickle
import struct ## new
import zlib
from PIL import Image, ImageOps

model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS.pth'))
model.eval()

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')

#Original

camera = cv.VideoCapture(0)

while cv.waitKey(1) & 0xFF != ord('q'):
    _, img = camera.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = img[y:y + h, x:x + w]
        faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
        # cv.imshow('Test', faceRegion)

        faceRegion = tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        mask, binary = model.forward(faceRegion)
        res = torch.mean(mask).item()
        # res = binary.item()
        print(res)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if res < 0.5:
            cv.putText(img, 'Fake', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        else:
            cv.putText(img, 'Real', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    cv.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)
    
    """

#IPcam

#start cam 沒用要註解掉 不然會一直等client傳送資料不會動作
HOST=''
PORT=8485

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
#end cam

camera = cv.VideoCapture(0)

while cv.waitKey(1) & 0xFF != ord('q'):
    #startcam
    while len(data) < payload_size:
        data += conn.recv(4096)
    # receive image row data form client socket
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # unpack image using pickle
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv.imdecode(frame, cv.IMREAD_COLOR)
    #end cam
    img = frame
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = img[y:y + h, x:x + w]
        faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
        # cv.imshow('Test', faceRegion)

        faceRegion = tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        mask, binary = model.forward(faceRegion)
        res = torch.mean(mask).item()
        # res = binary.item()
        print(res)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if res < 0.5:
            cv.putText(img, 'Fake', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        else:
            cv.putText(img, 'Real', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    cv.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)
    """
