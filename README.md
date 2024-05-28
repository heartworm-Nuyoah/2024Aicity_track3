# AICITY2024_Track3
This repo includes solution for AICity2024 Challenge Track 3 -  Naturalistic Driving Action Recognition


![framework.png](../_resources/framework-1.png)



# Getting Started

This page provides basic tutorials about the usage of our method.


<!-- TOC -->

## Datasets

The folder structure:

```
/xxxx
 / AICITY/datasets
  ├──data
    ├── A1_frame
    │   ├── VIDEO1
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── VIDEO2
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── ...
    ├── A2_frame
    │   ├── VIDEO1
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── VIDEO2
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── ...
    │   ├── ...
    ├── labels
    ├── A1_x
    │   ├── user_id_xxxxx
    │   │   ├── VIDEO1.MP4
    │   │   ├── VIDEO2.MP4
    │   │   ├── ...
    │   ├── ...
    ├── ...
    ├── A2
    │   ├── user_id_xxxxx
    │   │   ├── VIDEO1.MP4
    │   │   ├── VIDEO2.MP4
    │   │   ├── ...
    │   ├── ...
    ├── video_ids.csv
 

```
## Setup Environment
```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
```

## Preprocessing

```shell
# divide frame from videos and generate labels
python data_process.py 
```

## Training

```shell
# train the data set respectively in three perspectives
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh ./configs/Aicity/swin_base_patch244_window877_kinetics400_1k_total.py 4` --work-dir ./work_dir/train/total --validate

CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh ./configs/Aicity/swin_base_patch244_window877_kinetics400_1k_rear.py 4 --work-dir ./work_dir/train/rear --validate

CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh ./configs/Aicity/swin_base_patch244_window877_kinetics400_1k_dash.py 4 --work-dir ./work_dir/train/dash --validate

CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh ./configs/Aicity/swin_base_patch244_window877_kinetics400_1k_right.py 4 --work-dir ./work_dir/train/right --validate

```

## Inference

```shell
python tools/batch_inference.py --config ./work_dir/train/rear/swin_base_patch244_window877_kinetics400_1k.py --checkpoint ./work_dir/train/rear/epoch_9.pth --video_data_root /xxxx /datasets/data/A2_frame --label ./label_map.txt --step 4 --view rear

python tools/batch_inference.py --config ./work_dir/train/dash/swin_base_patch244_window877_kinetics400_1k.py --checkpoint ./work_dir/train/dash/epoch_6.pth --video_data_root /xxxx /datasets/data/A2_frame --label ./label_map.txt --step 4 --view dash

python tools/batch_inference.py --config ./work_dir/train/right1/swin_base_patch244_window877_kinetics400_1k.py --checkpoint ./work_dir/train/right1/epoch_4.pth --video_data_root /xxxx /datasets/data/A2_frame --label ./label_map.txt --step 4 --view right
```

## Post process

```shell
python post_process.py 
python further_process.py 
Finally, the results can be seen in results_submission_process.txt.
```



## Inference with Pre-Trained Models

### Test a dataset
 You can use the following commands to test a dataset. 

**1.Preprocessing** 
- cd $BASE_DIR/preprocess
```shell
python data_process.py 
```

**2.Inference** 

Download weights from [here](https://drive.google.com/drive/folders/1Hqzm1ksPyZKA6L4twIHz8JahKzJWBrAb) and put them into $BASE_DIR/mmaction2-master/work_dir/train.



- cd $BASE_DIR/mmaction2-master
```shell
python tools/batch_inference.py --config ./work_dir/train/rear/swin_base_patch244_window877_kinetics400_1k.py --checkpoint ./work_dir/train/rear/epoch_9.pth --video_data_root /xxxx /datasets/data/A2_frame --label ./label_map.txt --step 4 --view rear

python tools/batch_inference.py --config ./work_dir/train/dash/swin_base_patch244_window877_kinetics400_1k.py --checkpoint ./work_dir/train/dash/epoch_6.pth --video_data_root /xxxx /datasets/data/A2_frame --label ./label_map.txt --step 4 --view dash

python tools/batch_inference.py --config ./work_dir/train/right/swin_base_patch244_window877_kinetics400_1k.py --checkpoint ./work_dir/train/right/epoch_4.pth --video_data_root /xxxx /datasets/data/A2_frame --label ./label_map.txt --step 4 --view right
```
Then, the inference results can be seen in [submit_results_source](https://github.com/heartworm-Nuyoah/2024Aicity_track3/tree/main/mmaction2-master/submit_results_source) 


**3.Post process**

- cd $BASE_DIR/post_process
```shell
python post_process.py 

python further_process.py 

```
The results of the first step can be seen in results_submission.txt. Finally, the results can be seen in results_submission_process.txt.
## Contact
xuyuehuan, js@bupt.edu.cn

