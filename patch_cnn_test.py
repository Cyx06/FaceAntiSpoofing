import sys
import cv2
import numpy as np
from PIL import Image
import torch

sys.path.append('../')

from patch_based_cnn import net_baesd_patch, patch_test_transform
from processing_utils import get_file_list, FaceDection
from model_develop_utils import deploy_base
from config_patch import args
import torchvision.transforms as ts
import datetime

# Import the required modules
from IPython.display import clear_output
import socket
import sys
import matplotlib.pyplot as plt
import pickle
import struct ## new
import zlib
from PIL import Image, ImageOps




class rgb2ycrcb(object):
    '''
    自定义transform处理,将rgb图像转ycrcb
    :param object:
    :return:
    '''

    def __call__(self, img):
        img_new = img.convert("YCbCr")
        return img_new


class RandomCrop(object):

    def __init__(self, size, seed):
        self.seed = seed
        self.size = (int(size), int(size))

    def get_params(self, img, output_size):
        img = np.array(img)
        img_shape = img.shape
        w = img_shape[0]
        h = img_shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        np.random.seed(self.seed)
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        img = np.array(img)
        img_new = img[j:j + h, i:i + w]
        try:
            img_new = Image.fromarray(img_new.astype('uint8')).convert('RGB')
        except Exception as e:
            print("Image.fromarray(img.astype('uint8')).convert('RGB')")
            i, j, h, w = self.get_params(img, self.size)
        return img_new


def patch_cnn_single(model, face_detector, img, isface):
    '''

    :param model:
    :param face_detector:
    :param img:
    :param isface: the img is face img or not
    :return:
    '''
    if not isface:
        img = cv2.resize(img, (480, 640))
        # 人脸检测
        face_img = face_detector.face_detect(img)

    else:
        face_img = img

    if face_img is None:
        return None

    # 随机裁剪
    patch_size = 96
    img_Image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    data_len = 8
    seed_arr = np.arange(data_len)

    true_count = 0
    false_count = 0
    for i in range(data_len):
        img_transform = ts.Compose([RandomCrop(size=patch_size, seed=seed_arr[i])])
        try:
            img_patch = img_transform(img_Image)

            result_one = deploy_base(model=model, img=img_patch, transform=patch_test_transform)
            result_one=result_one[0]

            if result_one[0] > result_one[1]:
                false_count += 1
            else:
                true_count += 1
        except Exception as e:
            #print(e)
            continue

    # 集成判断
    #print("true_count", true_count, "false_count", false_count)
    if true_count >= false_count:
        return 1
    else:
        return 0



#original
"""

def patch_cnn_test(args, pre_path, test_dir, label, isface):
    '''

    :param args:
    :param pre_path: road to pretrain model
    :param test_dir: test img dir
    :param isface: img is face img or not, if not need to detect face
    :return:
    '''
    face_detector = FaceDection(model_name='cv2')
    model = net_baesd_patch(args)

    state_dict = torch.load(pre_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    time_begin = datetime.datetime.now()
    file_list = get_file_list(test_dir)
    count = 1
    true_num = 1
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue
        #here we call like "true_count 8 false_count 0"
        result = patch_cnn_single(model=model, face_detector=face_detector, img=img, isface=isface)

        if result is None:
            continue
        if result == label:
            count += 1
            true_num += 1
        else:
            print(file, "fake")
            count += 1
    print(count, true_num, true_num / count)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin
    print("time_all", time_all.total_seconds())


"""

#new for camera
def patch_cnn_test(args, pre_path, test_dir, label, isface):
    '''

    :param args:
    :param pre_path: road to pretrain model
    :param test_dir: test img dir
    :param isface: img is face img or not, if not need to detect face
    :return:
    '''
    face_detector = FaceDection(model_name='cv2')
    model = net_baesd_patch(args)

    state_dict = torch.load(pre_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    time_begin = datetime.datetime.now()
    file = cv2.VideoCapture(0)
    count = 1
    true_num = 1
    while cv2.waitKey(1) & 0xFF != ord('q'):

        _, img = file.read()
        if img is None:
            continue
        #here we call like "true_count 8 false_count 0"
        result = patch_cnn_single(model=model, face_detector=face_detector, img=img, isface=isface)

        if result is None:
            continue

        if result == label:
            print(1)
        else:
            print(0)

        cv2.imshow('CNN-Patch', img)
"""

#IPcam
def patch_cnn_test(args, pre_path, test_dir, label, isface):
    '''

    :param args:
    :param pre_path: road to pretrain model
    :param test_dir: test img dir
    :param isface: img is face img or not, if not need to detect face
    :return:
    '''
    # start cam 沒用要註解掉 不然會一直等client傳送資料不會動作

    HOST = ''
    PORT = 8485

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    # end cam

    face_detector = FaceDection(model_name='cv2')
    model = net_baesd_patch(args)

    state_dict = torch.load(pre_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    time_begin = datetime.datetime.now()
    file = cv2.VideoCapture(0)
    count = 1
    true_num = 1
    while cv2.waitKey(1) & 0xFF != ord('q'):

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
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        img = frame
        if img is None:
            continue
        #here we call like "true_count 8 false_count 0"
        result = patch_cnn_single(model=model, face_detector=face_detector, img=img, isface=isface)

        if result is None:
            continue

        if result == label:
            print(1)
        else:
            print(0)

        cv2.imshow('CNN-Patch', img)
"""
if __name__ == '__main__':
    test_dir = "testss"
    label = 1
    pre_path = "output/models/patch_fasd.pth"
    isface = True
    patch_cnn_test(args, pre_path=pre_path, test_dir=test_dir, label=label,
                   isface=isface)
