import os
import cv2
import numpy as np
import random
from munkres import Munkres
from skimage import io, color
from skimage import data, img_as_float
from skimage.transform import resize
from skimage.color import deltaE_cie76
import multiprocessing as mp
import math

ivdis=[]
M=[]

def _dir_is_image(i):
    suffix=os.path.basename(i).split('.')[-1]
    return suffix in ['jpg','jpeg','png']


def _video_dynamic(path):
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prvs = cv2.resize(prvs, (270, 480), interpolation=cv2.INTER_CUBIC)

    all_his = []
    while(1):
        ret, frame2 = cap.read()

        if not ret:
            break

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        next = cv2.resize(next, (270, 480), interpolation=cv2.INTER_CUBIC)
        flow_next = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        all_his.append(np.sqrt(np.sum(flow_next**2)))
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
    return np.sum(all_his)/len(all_his)

def cal(error):
    print(error)

'''
def _image_image_dissim(i,j):
    img0=data.load(i)
    img1=data.load(j)
    img0 = resize(img0, (224, 224))
    img1 = resize(img1, (224, 224))
    img0 = img_as_float(img0)
    img1 = img_as_float(img1)
    similarity = ssim(img0, img1, data_range=max(img0.max(),img1.max()) - min(img0.min(),img1.min()),multichannel=True)
    return 1-similarity

'''
def wtf(error):
    print(error)

def _image_image_dissim(i,j):

    img0 = io.imread(i)
    img1 = io.imread(j)
    img0 = resize(img0, (224, 224))
    img0 = color.rgb2lab(img0)
    img1 = resize(img1, (224, 224))
    img1 = color.rgb2lab(img1)
    diff = deltaE_cie76(img0, img1)
    res = sum(sum(diff**2))/(224*224)
    return res

def _image_video_dissim_par(img0,j):

    #img1 = io.imread(j)
    img1 = resize(j, (224, 224))
    img1 = color.rgb2lab(img1)
    diff = deltaE_cie76(img0, img1)
    res = sum(sum(diff**2))/(224*224)
    return res


def collect1(result):
    global ivdis
    ivdis.append(result)


def _video_video_dissim_par(img0,lis):
    k=[]
    for j in list(range(len(lis))):
        diff = deltaE_cie76(img0, lis[j])
        res = sum(sum(diff ** 2)) / (224 * 224)
        k.append(res)
    return k


def collect2(result):
    global M
    M.append(result)

'''
def _image_video_dissim(i,j):
     global ivdis

     img0=io.imread(i)
     ldiff=100000000
     img0 = resize(img0, (224, 224))
     img0 = color.rgb2lab(img0)
     
     ivdis=[]

     pool = mp.Pool()
   
         # Step 2: parallel by row
     for frame in  _video_frames(j):
         pool.apply_async( _image_video_dissim_par, args=(img0,frame), callback=collect1,error_callback=wtf)

     # Step 3: Don't forget to close
     pool.close()
     pool.join()

     return min(ivdis)
    '''






def _image_video_dissim(i, j):
    img0 = io.imread(i)
    ldiff = math.inf
    img0 = resize(img0, (224, 224))
    img0 = color.rgb2lab(img0)

    for frame in _video_frames(j):
        img1 = resize(frame, (224, 224))
        img1 = color.rgb2lab(img1)

        diff = deltaE_cie76(img0, img1)
        res = sum(sum(diff ** 2)) / (224 * 224)

        if ldiff > res:
            ldiff = res

    return ldiff



'''
def _image_video_dissim(i,j):
    # img0=data.load(i)
    # mst_sim=0
    # for frame in _video_frames(j):
    #     img1= frame
    #     img0 = resize(img0, (224, 224))
    #     img1 = resize(img1, (224, 224))
    #     img0 = img_as_float(img0)
    #     img1 = img_as_float(img1)
    #     similarity = ssim(img0, img1, data_range=max(img0.max(),img1.max()) - min(img0.min(),img1.min()),multichannel=True)
    #     if mst_sim<similarity:
    #         mst_sim=similarity
    # return 1-mst_sim
    return 0
'''

def _video_video_dissim(i,j):
    global M
    fr_list1=[]
    fr_list2=[]
    for m in _video_frames(i):
        fr_list1.append(m)
    for n in _video_frames(j):
        fr_list2.append(n)
    if len(fr_list1) > len(fr_list2):
        list1 = [fr_list1[i] for i in random.sample(range(len(fr_list1)), len(fr_list2))]  # list to seq?
        fr_list1 = list1
    elif len(fr_list1) < len(fr_list2):
        list2 = [fr_list2[i] for i in random.sample(range(len(fr_list2)), len(fr_list1))]  # list to seq?
        fr_list2 = list2

    length=len(fr_list1)
    M = []

    # tranform them
    for i in list(range(length)):
        img1=resize(fr_list1[i],(224,224))
        fr_list1[i]=color.rgb2lab(img1)
        img0 = resize(fr_list2[i], (224, 224))
        fr_list2[i] = color.rgb2lab(img0)


    pool = mp.Pool(8)

    # Step 2: parallel by row
    for i in list(range(length)):
        pool.apply_async(_video_video_dissim_par, args=(fr_list1[i], fr_list2), callback=collect2,error_callback=cal)
    # reordering of rows don't change the result of Hungarian algorithm
    pool.close()
    pool.join()

    m = Munkres()
    indexes = m.compute(M)
    total = 0
    for row, column in indexes:
        value = M[row][column]
        total += value

    return total/length

'''
def _video_frames(video_dir):
    vidcap = cv2.VideoCapture(video_dir)
    
    success, image = vidcap.read()
    # cv2.cvtColor(image,image, cv2.COLOR_BGR2Lab)
    while success:
        yield image
        success, image = vidcap.read()
'''


def _video_frames(video_dir):
    vidcap = cv2.VideoCapture(video_dir)

    success, image = vidcap.read()
    # cv2.cvtColor(image,image, cv2.COLOR_BGR2Lab)
    while success:
        su,im=vidcap.read()
        if(su):
            yield im
        success, image = vidcap.read()
