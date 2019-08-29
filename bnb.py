import math
import cv2
import numpy as np
import utils
import multiprocessing as mp
import heapq



results=[]
known_sim=[]
dynamic_list=[]
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


def get_permutation(N,M):
    from itertools import permutations

    return permutations(list(range(N)), min(N,M))



def collect_result(result):
    global results
    results.append(result)


def get_score(seq):
    plot = plot_penalty(seq)
    dis = get_dis(seq)
    sc = 0.5 * plot + 0.5 * dis
    #print(seq,sc)
    return  sc


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
    minis = heapq.nsmallest((min(8, length) - len(seq) - 1), flattened_D)  # (N-n)m in lower-bound  same if seq same
    seq.append(index)
    cost_already = get_score(seq)
    mini = 0.5 * sum(minis) + cost_already # first item the same for the same seq

    # find a upper bound
    lis=[]
    num= index
    flag = num
    sum1=0
    upper_seq = seq.copy()

    while len(upper_seq) < min(8,length):
        for j in range(num):
            lis.append(D[j][num])
        while flag < length:
            lis.append(D[num][flag])
            flag += 1

        for i in range(len(lis)):
            if i in upper_seq:
                lis[i] = math.inf
                continue
            if dynamic_list[i] < dynamic_list[num]:
                lis[i] += dynamic_list[num] - dynamic_list[i]

        les = min(lis)
        ind = lis.index(les)

        sum1 += les
        upper_seq.append(ind)
        num = ind
        flag=num
        lis=[]

    sum1=0.5*sum1+cost_already

    return sum1, mini,upper_seq
import glob,os
def get_all_visual_materials(material_dir):
    image_video_dir=glob.glob(os.path.join(material_dir,'*'))
    image_video_dir=sorted([i for i in image_video_dir if os.path.isdir(i)])
    return glob.glob(os.path.join(image_video_dir[0],'*.jpg'))

def get_kw_videos(material_dir):
	 print(material_dir)
	 video_dir=glob.glob(os.path.join(material_dir,'videos/*_*.mp4'))
	 image_dir=glob.glob(os.path.join(material_dir,'[0-9]*/*.jpg'))
	 #print('images',image_dir)
	 #print('videos',video_dir)
	 image_dir=[i for i in image_dir if 'frame' not in i]
	 if len(video_dir)==0:
	 	return image_dir
	 else:
	 	image_dir.extend(video_dir)
	 	return image_dir
    

if __name__=='__main__':


    # file = glob.glob("C:/Users/lily/Downloads/aliwood_product_dataset/aliwood_product_dataset/*")[1:]
    file = glob.glob('/home/liuchang/new_material/*')
    file = [i for i in file if os.path.isdir(i)]

    import os.path
    import pickle
    import sys

    M=8

    for material_dir in file:

         results=[]
         dynamic_list=[]

         id = material_dir.split('/')[-1][:12]
         print("id=", id)
         file_list = get_kw_videos(material_dir)
         print(file_list)
         dynamic_list = get_dynamic_score(file_list)

         bst_seq = []

         N = len(file_list)
         print('N',N)
         known_sim=np.zeros((N,N))
         dissim(file_list)
         L1 = []
         Lb = []
         Ub = []
         
         
         for n in range(N):
             L1.append([n])
         
         
         #L1=list(range(N))

         print(len(L1))
         count=0
         Lold = L1[:]
         upper_seq = []

         while True:

             Lb = []
             Ub = []
             L1 = []

             for seq in Lold:

                 for nxt in list(range(N)):
                     if nxt not in seq:
                         se1=seq.copy()
                         se1.append(nxt)
                         L1.append(se1)
                         d=known_sim.copy()
                         up,lo,upper_seq = sub_space(nxt, d, seq.copy(), N)
                         Lb.append(lo)
                         Ub.append(up)

             minUp = min(Ub)
             Lold=[]
             for i in range(len(L1)):
                 if Lb[i] <= minUp:
                     Lold.append(L1[i])

             print(len(Lold))
             count+=1
             if count==min(N-1,M-1):
             	break
             '''
             if(len(Lold)==1):
                Lold = upper_seq
             '''
         '''
         res=list(map(lambda x:get_score(x),Lold))
         inde = res.index(min(res))
         '''
         print(Lold[0])
         print([file_list[i] for i in Lold[0]])
        #  with open(id, 'wb') as f:
        #     pickle.dump([file_list[i] for i in Lold[0]], f)