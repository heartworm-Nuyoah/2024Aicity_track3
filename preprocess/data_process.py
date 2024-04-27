# <<<<<<< HEAD
# -*- coding: utf-8 -*-
# @File    : data_process.py
# @Brief   : build train, validate and inference dataset.

import os
import shutil
import cv2
import pandas
import numpy as np
import argparse
import glob
import os.path as osp
import sys
import warnings
from multiprocessing import Pool
import random
import time
from tqdm import tqdm, trange
import json
import re

def rmdir(pdir):
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
        print("{} path has been delete.".format(pdir))
    else:
        print('{} column does not exist.'.format(pdir))

def mkdir(pdir):
    if os.path.isdir(pdir):
        print("{} path has existed.".format(pdir))
    else:
        os.mkdir(pdir)
        print('{} path has been built.'.format(pdir))

appearance={'None'}


# classes = {
# '0': 'Normal Forward Driving'
# 1	'Drinking'
# 2	Phone Call(right)
# 3	Phone Call(left)
# 4	Eating
# 5	Text (Right)
# 6	Text (Left)
# 7	Reaching behind
# 8	Adjust control panel
# 9	Pick up from floor (Driver)
# 10	Pick up from floor (Passenger)
# 11	Talk to passenger at the right
# 12	Talk to passenger at backseat
# 13	yawning
# 14	Hand on head
# 15	Singing or dancing with music

# }


def get_userids(data_root):
    """
    param data_root: data root dir.
    return: user id list
    """
    userids=[]
    for dir in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root,dir)):
            userids.append(dir.split('_')[-1])
    return userids

def timecvt(str):
    """
    param str: time string'x:xx:xx'
    return: represent for time in second.
    """
    return float(int(str.split(':')[-2])*60+int(str.split(':')[-1]))


def div_frame(video_path, output_img_path):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.split(video_path)[1]
    # videoFPS=cap.get(cv2.CAP_PROP_FPS)
    # print (videoFPS)
    cnt = 0
    print("start")
    print(video_name)
    ind = 0
    nnn=0
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    t = time.time()
    selected_id=[2,5,9,13,16,20,24,27]
    while(True):#TODO
        ret, frame = cap.read()
        # show a frame
        if ret is True:
            cnt = cnt + 1
            if cnt in selected_id:
                cropped = frame
                output_name = os.path.join(output_img_path, 'frame{:>06d}.jpg'.format(ind))
                ind += 1
                cv2.imwrite(output_name, cropped)
            if cnt == 30:
                cnt = 0
                
        else:
            break
        print('\r', "{:s}[".format(os.path.split(video_path)[1]) + ">" * int(nnn / num_frames * 25) + "." * (
                    25 - int(nnn / num_frames * 25)) + "]{:.2f}s : {:.2f}% has been done.".format(
            time.time() - t, nnn / num_frames * 100), end='', flush=True)
        nnn+=1
    print("finished")
    cap.release()

    return

def batch_div_video2frames(data_root, data_div_root):
    # get user_id list
    user_ids = os.listdir(data_root)
    user_ids = list(filter(lambda x: x.startswith("user_id"), user_ids))
    # traversal user_id
    for user_id in user_ids:
        user_path = os.path.join(data_root,user_id)
        video_files = os.listdir(user_path)
        video_files = list(filter(lambda x :x.endswith('.MP4'), video_files))
        # traversal user_id's videos
        for video_file in tqdm(video_files):
            # extract frames from the videos to refered direction.
            video_path = os.path.join(user_path, video_file)
            output_img_dir = os.path.join(data_div_root,video_file.split('.MP4')[0])
            # rmdir(output_img_dir)
            mkdir(output_img_dir)
            div_frame(video_path, output_img_dir)

def filename_lowertext(data_root):
    """
    Lower text of the video file name and the csv file's videofile column.
    :param data_root:
    :return:
    """
    for userid in os.listdir(data_root):
        if not userid.endswith('.csv'):
            dirs = os.listdir(os.path.join(data_root,userid))
            videodirs = list(filter(lambda x: x.endswith((".MP4")), dirs))
            for video in videodirs:
                os.rename(os.path.join(data_root, userid, video), os.path.join(data_root, userid, video.split('.MP4')[0].lower()+'.MP4'))
            csvfiles = list(filter(lambda x: x.endswith((".csv")), dirs))
            for csvfile in csvfiles:
                if not csvfile.endswith('_lower.csv'):
                    csv_path = os.path.join(data_root, userid, csvfile)
                    df = pandas.read_csv(csv_path)
                    for value in df.values:
                        if type(value[1]) is str and value[1] != ' ':
                            if value[1][0]!=' ':
                                value[1]=value[1].lower()
                            else:
                                value[1]=''.join(list(value[1])[value[1].index(' ')+1:]).lower()
                    df.to_csv(os.path.join(os.path.split(csv_path)[0],os.path.split(csv_path)[1].split('.csv')[0].lower()+'_lower.csv'), encoding="utf-8-sig", header=True,index=False)
    return
def create_key(video):
    video=video.lower()
    if(video.lower().startswith('rear')):
        keyy='rear_'+video.replace("_noaudio","").split('_')[-2]+'_'+video.split('_')[-1]
    elif(video.lower().startswith('righ')):
        keyy='righ_'+video.replace("_noaudio","").split('_')[-2]+'_'+video.split('_')[-1]
    else:
        keyy='dash_'+video.replace("_noaudio","").split('_')[-2]+'_'+video.split('_')[-1]
    return keyy
def batch_csv2json(csv_label_root):
    """
    convert the csv files to json file.
    :return:
    """
    out_dict=dict()
    files = os.listdir(csv_label_root)
    csv_files = list(filter(lambda x: x.endswith((".csv")), files))
    a=0
    for csv_file in csv_files:
        if csv_file.endswith('.csv'):
            print(csv_file)
            a=a+1
            path=os.path.join(csv_label_root,csv_file)
            df = pandas.read_csv(path)

            for value in df.values:
                '''
                if type(value[0]) is str and value[0] !=' ':
                    video = value[0].lower()
                    out_dict[create_key(video)]=dict()
                    out_dict[create_key(video)][(value[5].split(' ')[1])] = dict(begin=value[3], end=value[4])
                '''
                '''
                if value[0] == ' ':
                    print(csv_file,'block')
                if type(value[0]) is str and value[0] != ' ':
                    if re.search(r'\d+', csv_file).group()!=re.search(r'\d+', value[0]).group():
                        print(csv_file,'mismatching')
                '''
                if type(value[0]) is str and value[0] != ' ':
                    video = value[0].lower()
                    if create_key(video) not in out_dict:
                        out_dict[create_key(video)] = {}

                    label = value[5].split(' ')[1]
                    if label not in out_dict[create_key(video)]:
                        out_dict[create_key(video)][label] =dict(begin=value[3], end=value[4])

            print(len(out_dict))


            # for value in df.values:
            #     if type(value[1]) is str and value[1] != ' ':
            #         video = value[1].lower()
            #         out_dict[video]=dict()
            #     if (type(value[6]) is int) or (type(value[6]) is float and not np.isnan(value[6])) or (type(value[6]) is str and value[6] !='N\A' and value[6] !='NA' and value[6] != 'NA ' ):
            #         out_dict[video][str(int(value[6]))] = dict(begin=value[4],end=value[5])
            #     else:
            #         print(value)
    print(a)
    print(len(out_dict))
    import json
    with open(os.path.join(csv_label_root,'labels.json'), "w") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
    return

def background_labels_by_id(data_root, det_root, val_id=['35133'], step = 4, interp1=[-8,24], interp0=[8,-8], result_file='add24_8_background_labels_binary.txt'):

    """
    divide the train data and validate data by val_user_id.
    :param data_root:
    :param det_root:
    :param result_file:
    :return:
    """
    result_file=str(step)+"_"+result_file

    def load_json_file(file):
        import json
        with open(file, "r") as f:
            dicts = json.load(f)
        return dicts
    dict = load_json_file(r'/disk1/xyh/AICITY/datasets/data/labels.json')
    data_root = os.path.join(os.path.split(data_root)[0], 'labels')  
    with open(os.path.join(data_root, 'train_'+val_id[0] +'_'+result_file), 'w') as f:
        f.writelines('')
    with open(os.path.join(data_root, 'val_'+val_id[0] +'_'+ result_file), 'w') as f:
        f.writelines('')
    for video_frames_dir in tqdm(os.listdir(det_root)):   
        num_frames = len(os.listdir(os.path.join(det_root, video_frames_dir)))  
        video_frames_path = os.path.join(det_root, video_frames_dir)  

        video = os.path.split(video_frames_path)[1] 
        begin_frame=0
        keyy=create_key(video)
        sss = 8
        for key in dict[keyy].keys():
            end_frame = min(int(timecvt(dict[keyy][key]['begin'])*8)+interp0[1], num_frames-sss)
            if begin_frame < end_frame-16:# limit the activity  happened time over 2s.

                for i in range(begin_frame, end_frame-(sss-step), step):
                    # try:
                    #     bbox = [min(bboxes[i:i+16, 1]),
                    #             min(bboxes[i:i+16, 2]),
                    #             max(bboxes[i:i+16, 3]),
                    #             max(bboxes[i:i+16, 4])]
                    # except:
                    #     print(i)
                    # bbox = [int(x) for x in bbox]
                    if keyy.split('_')[-2] in val_id:
                        with open(os.path.join(data_root, 'val_' + val_id[0] + '_' + result_file), 'a') as f:
                            f.writelines("{:s} {:d} {:d}\n".format(video, i, int(16)))###TODO 0
                    else:
                        with open(os.path.join(data_root, 'train_' + val_id[0] + '_' + result_file), 'a') as f:
                            f.writelines("{:s} {:d} {:d}\n".format(video, i,int(16)))###TODO  0
            begin_frame=int(timecvt(dict[keyy][key]['begin'])*8)+interp1[0]
            end_frame=min(int(timecvt(dict[keyy][key]['end'])*8)+interp1[1],num_frames-sss)

            for i in range(begin_frame,end_frame-(sss-step),step):
                # try:
                #     bbox = [min(bboxes[i:i+16, 1]),
                #             min(bboxes[i:i+16, 2]),
                #             max(bboxes[i:i+16, 3]),
                #             max(bboxes[i:i+16, 4])]
                # except:
                #     print(i)
                # bbox = [int(x) for x in bbox]
                if keyy.split('_')[-2] in val_id:
                    with open(os.path.join(data_root, 'val_'+val_id[0] +'_'+ result_file), 'a') as f:
                        f.writelines("{:s} {:d} {:d}\n".format(video, i, int(key)))###TODO1
                else:
                    with open(os.path.join(data_root, 'train_'+val_id[0] +'_'+result_file), 'a') as f:
                        f.writelines("{:s} {:d} {:d}\n".format(video, i, int(key)))###TODO1
            begin_frame = int(timecvt(dict[keyy][key]['end']) * 8)+interp1[1]+interp0[0]
    return

def change_anno(source_labels_file, out_anno_file, custom_func = None, shuffle = False ,begin ='right'):
  
    """
    Generate annotations.txt from labels.txt
    """
    with open(source_labels_file, 'r') as f:
        lines = f.readlines()
    lines = [x.split('\n')[0] for x in lines]

    if shuffle:
        random.shuffle(lines)
    assert custom_func is not None
    with open(out_anno_file, 'w') as f:
        for line in tqdm(lines):
            custom_func(f,line,begin=begin)
    return

def train_data_process():

    # divide frames
    video_root = r'/disk1/xyh/AICITY/datasets/data/A1'
    data_root = r'/disk1/xyh/AICITY/datasets/data/A1_frame'
    det_root = r'/disk1/xyh/AICITY/datasets/data/A1_frame'

    str_result_total  = 'total_labels.txt'
    def custom_func(f, line, begin='right'):
            """
            customed annotation format write function
            :param f: file bar
            :param line: original label string line
            :return:
            """
            # desolve the original label line
            line = line.split(' ')
            video = line[0]
            begin_frame_id = int(line[1])
            bbox = line[2].split('_')
            bbox = [int(x) for x in bbox]
            clip_size = 16  # line[-2]
            label = int(line[-1])
            if video.lower().startswith(begin) and label != 16:
                # write new annotation file
                f.writelines(
                    # "{:s} {:d} {:d}_{:d}_{:d}_{:d} {:d} {:d}\n".format(video, begin_frame_id, bbox[0], bbox[1], bbox[2],
                    #                                                    bbox[3], clip_size, label))
                    "{:s} {:d} {:d} {:d}\n".format(video, begin_frame_id, clip_size, label))  # annotations format
            return

    #mkdir(data_root)
    #batch_div_video2frames(video_root, data_root)

    ## lower the text of file names.
    #filename_lowertext(video_root)

    # convert csv files into a json file.
    # csv_label_root = r'/disk1/xyh/AICITY/datasets/data/labels/A1'
    # batch_csv2json(csv_label_root)

    # build train data and validate data.
    val_id_split_list = [['99882','99660','99635','98389','98067','96371','96269','93542','93491']]

    step = 8
    for val_id in val_id_split_list:
        #background_labels_by_id(video_root, det_root, val_id=val_id, step=8, interp1=[0,0], interp0=[0,0],result_file=str_result_total)
        background_labels_by_id(video_root, det_root, val_id=val_id, step=8, interp1=[8, -8], interp0=[8, -8],result_file=str_result_total)
 
        # read xxx_labels.txt and build anno_xxx.txt
        begin_list = ['rear','dash','right']
        for begin in begin_list:
            # # trian dataset
            source_labels_file = '/disk1/xyh/AICITY/datasets/data/labels/' +'train_'+ val_id[0] +'_' + str(step) +'_'+ str_result_total 
            out_anno_file = os.path.join(r'/disk1/xyh/AICITY/datasets/data/labels', begin + '_anno_train.txt')
            change_anno(source_labels_file, out_anno_file, custom_func, shuffle=False, begin=begin)
            # validate dataset
            source_labels_file =  '/disk1/xyh/AICITY/datasets/data/labels/' +'val_'+ val_id[0] +'_' + str(step) +'_'+ str_result_total 
            out_anno_file = os.path.join(r'/disk1/xyh/AICITY/datasets/data/labels', begin + '_anno_val.txt')
            change_anno(source_labels_file, out_anno_file, custom_func, shuffle=False, begin=begin)

def test_data_process():
    # divide frames
    video_root = r'/disk1/xyh/AICITY/datasets/data/A2/SET-A2'
    data_root = r'/disk1/xyh/AICITY/datasets/data/A2_frame'
    mkdir(data_root)
    batch_div_video2frames(video_root, data_root)


if __name__ == '__main__':
    train_data_process() # for train A1
    # test_data_process() # for test A2
    # print(1)




