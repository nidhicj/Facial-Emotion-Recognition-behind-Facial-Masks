'''
Returns 20-landmarks points along with image and label for CK+ dataset   
'''

import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pickle as pkl
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid
ImageFile.LOAD_TRUNCATED_IAMGES = True

import random as rd 
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from numpy import linspace
from matplotlib import cm
import math


ImageFile.LOAD_TRUNCATED_IAMGES = True
def PIL_loader(path):
    try:
        # print(path)
        with open(path, 'rb') as f:
            # print('Verified')
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
def get_class(idx):
        classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

        return classes[idx]
def change_emotion_label_same_as_affectnet(emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 0
        elif emo_to_return == 1:
            emo_to_return = 6
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 4
        elif emo_to_return == 4:
            emo_to_return = 1
        elif emo_to_return == 5:
            emo_to_return = 2
        elif emo_to_return == 6:
            emo_to_return = 3

        return emo_to_return
'''

CKplus dataset labels
0=neutral
1=anger
2=contempt
3=disgust
4=fear
5=happy
6=sadness
7=surprise

My Labels:
0: Neutral
1: Happy
2: Angry
3: Surprise
4: Sad
5: Fear
6: Disgust

'''
def default_reader(fileList, landmarksfile,num_classes):
    
    imgList = []

    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0
        
    one_list = []
    with open(fileList, 'r') as fp:
        for line in fp.readlines(): 
            one_list.append(line)

    with open(landmarksfile, 'rb') as handle:
        unserialized_data = pkl.load(handle)

    for i,img_path in enumerate(one_list):
        img_name = (img_path.rsplit('/',1)[1]).strip()
        img_path = img_path.strip()
        if img_path in unserialized_data:
            if unserialized_data[img_path][1] == 2:
                continue
            elif len(unserialized_data[img_path][1]) != 0:
                # print(unserialized_data[img_name][1])
                exp , landmarks_68 = unserialized_data[img_path][1], unserialized_data[img_path][0]
                # print(exp)
                exp = exp[0] - 1
                exp = change_emotion_label_same_as_affectnet(exp)
                if num_classes ==4:
                    if exp == 4 or exp == 5 or exp == 6:
                        continue
                imgList.append([img_path, exp, landmarks_68])
                num_per_cls_dict[exp] = num_per_cls_dict[exp]+1
 
    print('Total included ', len(imgList), num_per_cls_dict)
    return imgList ,num_per_cls_dict
# import pickle as pkl
# rootfolder= '/netpool/work/gpu-3/users/joshini/datasets/Masked_images/CK+/masked_faces'
# fileList = '/projects/joshi/projects/new_ckplus_allImagesList.txt'
# landmarksfile='/projects/joshi/projects/new_ckplus_alllandmarksfile.pkl'
# imgList,num_per_cls_dict =  default_reader(fileList,landmarksfile, num_classes=8)
# imgPath, target_expression,landmarks_68 = imgList[0]
# # print(imgPath)
# imgPath, target_expression,landmarks_68 = imgList[0]
# img = PIL_loader(os.path.join(rootfolder, imgPath)) 
# # img = img.size([256,256])
# img.size[0]

def convert68to20(landmarks_68 ,input_imgsize, target_imgsize):
    '''
    a) Occlusion Aware Facial Expression Recognition Using CNN With Attention Mechanism
    b)  https://github.com/mysee1989/PG-CNN/blob/master/convert_point/pts68_24

    Out of 68 points : 20 are recomputed along with score as minimum of points considered
    '''
    # landmarks_68 = np.transpose(landmarks_68)
    #print(landmarks_68.shape) #np array of size 68x3
   
    landmarks_20 = []

    #16 standard landmark points from eyebrow, eyes, nose and mouth
    single_points = [17,19,21,22,24,26,36,39,42,45,27] 
    # 2-Point from left eye, 2 from right eye, next left cheek and right cheek = 6 points for averaging
    double_points = [
            [21,22],
            [21,39],
            [22,42],
            [37,41],
            [38,40],
            [43,47],
            [44,46],
            [19,37],
            [24,44]
            ]  

    # 2 more points at offfset from left mouth corner:49 and right mouth corner:55

    
    #First add 16
    for index in single_points:

        landmark = np.array(landmarks_68[index],dtype=np.float32)    
        # print([index-1],landmark)
        landmarks_20.append(landmark)
    
    # print("landmarkslist with single points: ",landmarks_24)
    # midpt_list = []
    #Add average 6    
    for ele in double_points:
        point1 = landmarks_68[ele[0]]#  [:2]
        # score1 = landmarks_68[ele[0]-1]#[2]
        point2 = landmarks_68[ele[1]]#[:2]
        # score2 = landmarks_68[ele[1]-1]#[2]
        
        # midpoint = np.round(np.mean(np.array([point1, point2])),3)  #.reshape(1,2)
        
        midpoint = np.array((np.mean([point1[0],point2[0]]) , np.mean([point1[1],point2[1]])),dtype=np.float32)
        # print([ele[0]-1],[ele[1]-1],point1,point2,midpoint)
        # midpt_list.append(midpoint)
        landmarks_20.append(midpoint)
    
    # print('midpoint: ',midpoint)

    # print(landmarks_24)
    landmarks_20 = np.asarray(landmarks_20, dtype= np.float32)  #.reshape(24,3)
    
    landmarks_20_scaled = landmarks_20 * target_imgsize  / (input_imgsize)
    
    
    # print(landmarks_24)
    # print('scaled: ',landmarks_24_scaled)
    return landmarks_20#landmarks_24

    
class ImageList(data.Dataset):
    def __init__(self, root, fileList, landmarksfile='../data/Jaffe/jaffe_landmarks_scores.pkl', num_classes=8, 
                                target_imgsize = 28,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = num_classes
        
        self.imgList, self.num_per_cls_dict =  list_reader(fileList,landmarksfile, self.cls_num)
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList

    def __getitem__(self, index):

        imgPath, target_expression,landmarks_68 = self.imgList[index]
        # print(imgPath)
        img = PIL_loader(imgPath)
        # print(img)
        landmarks = convert68to20(landmarks_68 ,img.size[0], target_imgsize=28)        
            
        landmarks_20 = [(int(landmarks[i][0]),int(landmarks[i][1])) for i in range(0,20)]
        
        if self.transform is not None:
            img = self.transform(img)

        return  img, target_expression ,torch.tensor(landmarks_20)

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

        
if __name__=='__main__':

   #get_subject_independent_fold_files()
   
   rootfolder= '/netpool/work/gpu-3/users/joshini/datasets/Masked_images/CK+/masked_faces'
   fileList = '/netpool/work/gpu-3/users/joshini/datasets/Masked_images/CK+/newckplus_trainList.txt'
   folds = 10
   classes = 8
   
      
   transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])
   dataset = ImageList(rootfolder, fileList,landmarksfile='/netpool/work/gpu-3/users/joshini/datasets/Masked_images/CK+/trainlandmarksfile.pkl',  transform=transform)

   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 2:
         #  print(data[0][0].size(), data[1],data[2], data[2].size())
          continue
       else:
          break

