import numpy as np
import os
import time
from scipy.special import softmax
import cv2
import pandas
import math
import csv
import ast
import sys
import json
from pprint import pprint
from collections import defaultdict, OrderedDict
from scipy.optimize import linear_sum_assignment
import argparse
from tqdm import tqdm
import shutil
import random
import matplotlib.pyplot as plt

CLASSES = [
'0:Normal Forward Driving',
'1:Drinking',
'2:Phone Call(right)',
'3:Phone Call(left)',
'4:Eating',
'5:Text (Right)',
'6:Text (Left)',
'7:Hair / makeup',
'8:Adjust control panel',
'9:Pick up from floor (Driver)',
'10:Pick up from floor (Passenger)',
'11:Talk to passenger at the right',
'12:Talk to passenger at backseat',
'13:yawning',
'14:Hand on head',
'15:shaking or dancing with music']

cls_id= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
def smoothing(x, k=2):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k-1
    '''

    ###method 1
    # score_x = x[:,1:]
    # smoothed_result = np.zeros_like(x)
    # l = len(score_x)
    # s = np.arange(-k, l - k)
    # e = np.arange(k, l + k)
    # s[s < 0] = 0
    # e[e >= l] = l - 1
    # y = np.zeros(score_x.shape)
    # for i in range(l):
    #     y[i] = np.mean(score_x[s[i-1]:e[i]], axis=0)
    #     smoothed_result[i,0] = x[i,0]
    #     smoothed_result[i,1:] = y[i]        
    ##method 2
    score_x = x[:,1:]
    smoothed_result = np.zeros_like(x)
    l = len(score_x)
    s = np.arange(-k, l - k)
    e = np.arange(k-1, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(score_x.shape)
    for i in range(l):
        y[i] = np.mean(score_x[s[i]:e[i]], axis=0)
        smoothed_result[i,0] = x[i,0]
        smoothed_result[i,1:] = y[i]    
    return smoothed_result 
def read_sigle_result_txt(result_txt_path):

    data = np.loadtxt(result_txt_path)
    for i in range(len(data)):
        data[i,1:] = softmax(data[i,1:])
    data =smoothing(data,k=2)
    return data

def read_csvfile(path):
    df = pandas.read_csv(path)

    video = {}
    for i ,value in enumerate (df.values) :
        video[i+1] = [value[1],value[2],value[3]]
    return video

def read_path(results_root,id_video,video_id_dict):
    dash_txt_name = list(filter(lambda x:x.startswith((video_id_dict[id_video][0].split('.')[0])),os.listdir(results_root[0])))[0]
    rear_txt_name = list(filter(lambda x:x.startswith((video_id_dict[id_video][1].split('.')[0])),os.listdir(results_root[1])))[0]
    right_txt_name = list(filter(lambda x:x.startswith((video_id_dict[id_video][2].split('.')[0])),os.listdir(results_root[2])))[0]
    # print(dash_txt_name)
    # print(rear_txt_name)
    # print(right_txt_name)
    dash_path = os.path.join(results_root[0],dash_txt_name)
    rear_path = os.path.join(results_root[1],rear_txt_name)
    right_path = os.path.join(results_root[2],right_txt_name)
    return dash_path,rear_path,right_path

def remove_begin_end(df_total):
    df_total = df_total.reset_index()

    print(len(df_total))
    remove_df_total = df_total
    for i in range(1,len(remove_df_total)):
        if df_total.loc[i][2]<df_total.loc[i-1][3]:
            if df_total.loc[i][0]==df_total.loc[i-1][0]:
                remove_df_total = remove_df_total.drop([i],axis=0)
                print(len(remove_df_total))
    
    # remove_df_total = remove_df_total.drop(columns=['index'])
    remove_df_total = remove_df_total.drop(columns=['index'])
    return remove_df_total

def ensenble_three_views(dash_path,rear_path,right_path):

    dash_result = read_sigle_result_txt(dash_path)
    rear_result = read_sigle_result_txt(rear_path)
    right_result = read_sigle_result_txt(right_path)

    N_f = min(len(dash_result),len(rear_result),len(right_result))
    ensenble_result = np.zeros((N_f,len(dash_result[0,:])))
    for i in range(N_f):
        ensenble_result[i,0] = rear_result[i,0]
        ensenble_result[i,1:] = np.maximum(np.maximum(dash_result[i,1:],rear_result[i,1:]),right_result[i,1:])
    return ensenble_result

def get_classification(sequence_class_prob):
    # classify=[[x,y] for x,y in zip(np.argmax(sequence_class_prob, axis=1),np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob[:,1:], axis=1) #returns list of position of max value in each list.
    probs= np.max(sequence_class_prob[:,1:], axis=1)  # return list of max value in each  list.
    return labels_index,probs

def activity_localization(prediction_smoothed):

    action_idx, action_probs =  get_classification(prediction_smoothed)
    threshold = each_cls_threshold(prediction_smoothed)
    action_tag = np.zeros_like(action_idx)
    for i in range(len(action_idx)):
        if action_probs[i]>threshold[action_idx[i]]:
            action_tag[i] = 1
    print('threshold:', threshold)
    activities_idx = []
    startings = []
    endings = []
    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_idx[i])
            start = i
            end = i+1
            startings.append(start)
            endings.append(end)
    # print('activities_idx', activities_idx)  
    # print('start', startings)
    # print('end', endings)
    return activities_idx,startings,endings ,threshold

def general_submission(data):
    data_filtered = data[data[1] != 16]
    # df_total = merge_and_remove(data_filtered)
    df_total = pandas.DataFrame([[0, 0, 0, 0]], columns=[0, 1, 2, 3])
    for i in range(1, 31):
        # print(i)
        data_video = data[data[0]==i]
        # print(data_video)
        list_label = data_video[1].unique()
        # print(list_label)
        for label in list_label:
            data_video_label = data_video[data_video[1]== label]
            data_video_label = data_video_label.reset_index()
            # print('data_video_label')
            # print(data_video_label)
            # if len(data_video_label) == 1 :
            #     continue
            for j in range(len(data_video_label)-1):
                if data_video_label.loc[j+1, 2] - data_video_label.loc[j, 3] <=8.5:
                    data_video_label.loc[j+1, 2] = data_video_label.loc[j, 2]
                    data_video_label.loc[j, 3] = 0
                    data_video_label.loc[j, 2] = 0
            # print(data_video_label)
            df_total = df_total.append(data_video_label)
    df_total = df_total[df_total[3]!=0]
    df_total = df_total[df_total[1]!=0]
    df_total = df_total[df_total[1]!=16]
    # df_total = df_total[df_total[3] - df_total[2] >3.5]
    df_total = df_total[df_total[3] - df_total[2] >3]
    df_total = df_total.drop(columns=['index'])
    df_total = df_total.sort_values(by=[0, 2])
    df_total[0] = df_total[0].map(lambda x: int(float(x)))
    df_total[1] = df_total[1].map(lambda x: int(float(x)))
    df_total[2] = df_total[2].map(lambda x: (int(math.ceil(float(x)))))
    df_total[3] = df_total[3].map(lambda x: int(math.ceil(float(x))))
    remove_df_total = remove_begin_end(df_total)

    remove_df_total.to_csv('/disk1/xyh/AICITY/2024code/mmaction2-master/submit_results_source/swin/results_submission.txt', sep=' ', index = False, header=False)
    return df_total
    # return  remove_df_total

def each_cls_threshold (prediction_smoothed):
    threshold = []
    for cls in cls_id:
        score_threshold = np.sort(prediction_smoothed[:,cls+1])[-15].mean()
        threshold.append(score_threshold)
    threshold = np.minimum(threshold, 0.97)
    return threshold

def proposal_result(result_root,csv_path):

    video_id_dict = read_csvfile(csv_path) 
    dataframe_list = []
    for id_video in video_id_dict:
        dash_path,rear_path,right_path = read_path(result_root,id_video,video_id_dict)
        ensenble_result = ensenble_three_views(dash_path,rear_path,right_path)
        activities_idx ,startings,endings,threshold = activity_localization(ensenble_result)
        for idx, s, e in zip(activities_idx, startings, endings):
            start = s * 0.5
            end = e * 0.5
            label = cls_id[idx]
            # print(
            #     '{}\t{}\t{:.1f}s - {:.1f}s\t'.format(id_video, label,start, end))
            dataframe_list.append([id_video, label,start, end])
    data_frame = pandas.DataFrame(dataframe_list, columns =[0, 1, 2, 3])
    df_total = general_submission(data_frame)      
    print(np.array(df_total))  
    return np.array(df_total)


if __name__ =='__main__':

    # dash/rear/right #
    csv_path = '/disk1/xyh/AICITY/datasets/data/2024-data_video_ids1.csv'
    results_root = [
                        [
                        '/disk1/xyh/AICITY/2024code/mmaction2-master/submit_results_source/swin/dash',
                        '/disk1/xyh/AICITY/2024code/mmaction2-master/submit_results_source/swin/rear',
                        '/disk1/xyh/AICITY/2024code/mmaction2-master/submit_results_source/swin/right']
                        ]

    proposal_result0 = proposal_result(results_root[0],csv_path)


