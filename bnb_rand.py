import math
import time
import random
from multiprocessing import pool
import cv2
import numpy as np
import utils
import multiprocessing as mp
import heapq
from random import shuffle
count=0
S=0
results=[]
known_sim=[]
lam=0.5
dynamic_list=[]
L=[]
def get_dynamic_score(file_list):
    ans = []
    for i in file_list:
        if not utils._dir_is_image(i):
            ans.append(utils._video_dynamic(i))
        else:
            ans.append(0)
    return np.array(ans)

def plot_penalty(A):
    global dynamic_list

    d=dynamic_list[A]
    #print(d)
    p_func=lambda x,y:x-y if y<x else 0
    return np.sum([p_func(d[i],d[i+1]) for i in range(len(d)-1)])

def errr(er):
    print(er)

def dissim(file_list):
    global known_sim

    for i in list(range(len(file_list))):
        for j in list(range(len(file_list))):
            if i >= j:
                known_sim[i][j]=math.inf # for future bnb
            elif not utils._dir_is_image(file_list[i]):
                if utils._dir_is_image(file_list[j]):#consider parallel differnt image against same video
                    known_sim[i][j] = utils._image_video_dissim(file_list[j], file_list[i])
                else:
                    known_sim[i][j] = utils._video_video_dissim(file_list[i], file_list[j])
            elif not utils._dir_is_image(file_list[j]):
                known_sim[i][j] = utils._image_video_dissim(file_list[i], file_list[j])
            else:
                known_sim[i][j] = utils._image_image_dissim(file_list[i], file_list[j])
    return

def get_dis(seq):
    global known_sim
    s = 0.0
    for i in list(range(len(seq)-1)):
        if seq[i] > seq[i+1]:
            s += known_sim[seq[i+1]][seq[i]]
        else:
            s += known_sim[seq[i]][seq[i + 1]]
    return s


def get_permutation(n,m):
    from itertools import permutations

    return permutations(list(range(n)), min(n,m))



def collect_result(result):
    global results
    results.append(result)


def __get_score(i):
    seq=random.sample(L,S)
    plot = plot_penalty(seq)
    dis = get_dis(seq)
    sc = 0.5 * plot + 0.5 * dis
    #print(seq,sc)
    
    return  seq,sc


def sub_space(index,D,seq,length):
    # subspace seq::index
    global dynamic_list
    # (N-n)m the same if seq the same
    if len(seq) != 0:
        for i in range(len(seq)):
            num = seq[i]
            flag = num + 1
            for j in range(num):
                D[j][num] = math.inf
            while flag < length:
                D[num][flag] = math.inf
                flag += 1
    flattened_D = [y for x in D for y in x]
    minis = heapq.nsmallest((min(S, length) - len(seq) - 1), flattened_D)  # (N-n)m in lower-bound  same if seq same
    seq.append(index)
    cost_already = get_score(seq)
    mini = lam * sum(minis) + cost_already # first item the same for the same seq

    # find a upper bound
    lis=[]
    num= index
    flag = num
    sum1=0
    upper_seq = seq.copy()

    while len(upper_seq) < min(S,length):
        for j in range(num):
            lis.append(lam*D[j][num])
        while flag < length:
            lis.append(lam*D[num][flag])
            flag += 1

        for i in range(len(lis)):
            if i in upper_seq:
                lis[i] = math.inf
                continue
            if dynamic_list[i] < dynamic_list[num]:
                lis[i] += (1-lam)*(dynamic_list[num] - dynamic_list[i])

        les = min(lis)
        ind = lis.index(les)

        sum1 += les
        upper_seq.append(ind)
        num = ind
        flag=num
        lis=[]

    sum1=sum1+cost_already

    return sum1, mini

'''
if __name__ == '__main__':
    lis=[3,2,5,6,2,1]
    mini=min(lis)

    print(lis.index(mini))
'''
def get_all_image(img_dir,need_sort=False):
    all_image=glob.glob(os.path.join(img_dir,'*.png'))+glob.glob(os.path.join(img_dir,'*.jpg'))
    if need_sort:
        return sorted(all_image)
    else:
        return all_image



    


if __name__=='__main__':
    import glob

    # file = glob.glob("C:/Users/lily/Downloads/aliwood_product_dataset/aliwood_product_dataset/*")[1:]
    file = glob.glob('/home/liuchang/TaobaoItem/results/*/')

    import os.path
    import pickle
    import sys
    import time
    import math
    worker=32
    pool=pool.Pool(worker)
    le_dic={}
    lfile=open('/home/liuchang/PA/length','rb')
    lfile = pickle.load(lfile)
    for i in range(len(lfile)//2):
      le_dic[lfile[2*i]] = lfile[2*i+1]

    for material_dir in file:
         results=[]
         dynamic_list=[]
         count=0
         id = material_dir.split('/')[-2]
         print(id)
         S = le_dic[material_dir]
         print(S)
         file_list = get_all_image(material_dir)
         
         file_list=[i for i in file_list if 'shop' not in i]
         start_time=time.time()
         
         dynamic_list = get_dynamic_score(file_list)

         bst_seq = []

         N = len(file_list)
         print(N)
         if N < S:
             print("N < S, SKIPPED")
             continue
         known_sim=np.zeros((N,N))
         dissim(file_list)
         
         sc=math.inf
         L = list(range(N))
         lis=[]
         
         t0=time.time()
         
         while time.time()-t0<10*60:
            for seq,seq_v in pool.imap_unordered(__get_score,range(worker)):
              count+=1
              if seq_v<sc:
                sc=seq_v
                lis=seq
       
         '''
         for i in range(1500000):
           L = list(range(N))
           shuffle(L)
           L1=L[:S]
           shuffle(L1)
           tmp=get_score(L1)
           if sc > tmp:
             sc = tmp
             lis=L1 
         '''

         #print("S=%d N=%d feature_time = %s ,bnb time = %s" %(S,N,feature_time-start_time,time.time()-feature_time))
         print(lis)
         print(sc)
         print(count)
         '''
         with open(id, 'wb') as f:
            pickle.dump([file_list[i] for i in lis], f)
         '''
         
