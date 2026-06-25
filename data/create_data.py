"""
Description: 生成训练数据集，数据集统计信息，以及数据版本更新信息
Author: 算法组 蔡雨霖
Date: 2024-05-11

"""

import os
import yaml
from datetime import datetime
from os.path import join
import numpy as np

#---------------------#
# 读取数据集数据数量
#---------------------#
def list_images(path, dirs):
    
    Dictionary = {}
    sample_sum = 0
    for dir in dirs:
        if dir.split('-')[0] == "GEOAI":
            labels = os.listdir(join(path, dir, "label"))
            Dictionary[f"{path.split('/')[-1]}/{dir}".ljust(90)] = len(labels)
            sample_sum = sample_sum + len(labels)

    return Dictionary, sample_sum


#--------------------#
# 生成数据集读取文件
#--------------------#
def list_folders(path, dirs):

    # 创建空数组
    train_files = []
    val_files   = []
    
    # 生成训练验证数据集读取文件
    dataset_quantity = 0
    for dir in dirs:
        dir_path = os.path.join(path, dir)
        if dir.split('-')[0] == "GEOAI":
            train_file_path = os.path.join(dir_path, 'train.txt')
            val_file_path = os.path.join(dir_path, 'val.txt')
            if os.path.exists(train_file_path) and os.path.exists(val_file_path):
                train_files.append(train_file_path)
                val_files.append(val_file_path)

            else:
                raise ValueError(f"There is no train/val.txt in dir # {dir} #, please check it!")

            dataset_quantity += 1
    
    return train_files, val_files, dataset_quantity


#----------------#
# 生成更新信息
#----------------#
def update(file, incharge, model_vision, Vision_Instruction, quantity, Change_dir):
    
    # 更新信息 
    infos = [
    f'## {model_vision} ##\n',
    f' # Date：{datetime.now()}',
    f' # Responsible Person：{incharge}',
    f' # Dataset Used: {quantity}',
    f' # Dataset update：{Change_dir}',
    f' # Update news：{Vision_Instruction}',
    ]
    
    for comment in infos:
        file.write(f'\n{comment}')

if __name__ == "__main__":

# 修改 ------------------------------------------------------------------------------------------------------------------#
    #--------------#
    # 负责人
    #---------------------------------------------#
    incharge = "yulin"                            # 负责人
    #---------------------------------------------#
    project = "ChangeD"
    model_vision = "1-ChangeDetect-2605-v1.2"           # 模型版本
    #---------------------------------------------#
    nc, names = 1, ["change"]
    #---------------------------------------------#
    # 该版本数据更新说明
    #---------------------------------------------#
    Vision_Instruction = "[无人机] 变化检测：增加负样本，城区的车辆和相关建筑废料等。"
    #------------------------#
    # 数据读取路径
    #------------------------#
    path = ["/home/yulin/0-data/1-ChangeDetct"]
#------------------------------------------------------------------------------------------------------------------------#


    #----------------------------------#
    # 分别统计指定数据集路径中的数据
    #----------------------------------#
    # 按顺序遍历文件夹
    DataItem, Dataset_Dictionary = {}, {}
    Train_files, Val_files = [], []
    DataSet_Quantity, Sample_Sum = 0, 0
    for p in path:
        dataitem = p.split('/')[-1]
        DataItem[dataitem] = os.listdir(p)
        DataItem[dataitem].sort()

        # 读取数据集 train.txt, val.txt 以及统计数据集个数
        train_files, val_files, dataset_quantity = list_folders(p, DataItem[dataitem])

        # 读取数据集的样本信息
        dataset_dictionary, sample_sum = list_images(p, DataItem[dataitem])

        # 形成训练集读取文件
        Train_files += train_files
        Val_files += val_files
        # 统计数据集的个数
        DataSet_Quantity += dataset_quantity
        # 统计训练样本总数量
        Sample_Sum += sample_sum
        # 统计每个数据集中的样本数量
        Dataset_Dictionary.update(dataset_dictionary)


    #------------------------------------------#
    # 更新训练的数据文件：0-{project}.yaml
    #------------------------------------------#

    data = {'nc': nc, 'names': names, 'train': Train_files, 'val': Val_files}
    with open(f'0-{project}.yaml', 'w') as file:

        file.write(f'#【Train dataset for {project}】\n')                        # 写入文件头信息
        file.write(f'# Dataset Used: {DataSet_Quantity}\n')
        file.write(f'# Dataset Sum：{Sample_Sum}\n')
        file.write(f'# Date：{datetime.now()}\n')
        file.write(f'\n\n')
        yaml.dump(data, file, default_flow_style=False, indent=4)       # 写入训练数据文件

    #-------------------------------------------#
    # 生成初始 1-Dataset-stat.yaml
    #-------------------------------------------#
    if not os.path.exists('1-Dataset-stat.yaml'):
        with open('1-Dataset-stat.yaml', 'w') as file:
            yaml.dump(Dataset_Dictionary, file, default_flow_style=False, indent=4)  # 统计数据集信息

    #--------------------------------------------#
    # 更新版本数据信息：2-Vision-info.yaml
    #--------------------------------------------#

    with open('1-Dataset-stat.yaml', 'r') as file:
        dataset_stat = yaml.safe_load(file)  # old

    Change_dir = []

    # 判断数据集有没有变动
    for key in Dataset_Dictionary:
        if key not in dataset_stat:
            Change_dir.append(key.strip())
        else:
            if Dataset_Dictionary[key] != dataset_stat[key]:
                Change_dir.append(key.strip())

    with open('2-Vision-info.yaml', 'a') as file:
        # file.write('# 【Dataset infos of BSD history vision】')
        file.write(f'\n\n')
        update(file, incharge, model_vision, Vision_Instruction, DataSet_Quantity, Change_dir)  # 更新信息


    #------------------------------------------#
    # 统计数据集信息：1-Dataset-stat.yaml
    #------------------------------------------#

    with open('1-Dataset-stat.yaml', 'w') as file:

        file.write(f'#【Statics of the {project} dataset】\n')  # 写入文件头信息
        file.write(f'# Dataset Used: {DataSet_Quantity}\n')
        file.write(f'# Dataset Sum：{Sample_Sum}\n')
        file.write(f'# Date：{datetime.now()}\n')
        file.write(f'\n\n')
        yaml.dump(Dataset_Dictionary, file, default_flow_style=False, indent=4)  # 统计数据集信息

    
        
    