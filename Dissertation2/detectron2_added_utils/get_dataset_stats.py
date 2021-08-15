from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from pathlib import Path
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def register_load_dataset(dataset_name,source):
    ann_path=source/'annotations'/'instances.json'
    imgs_path=source/'images'
    try :
        register_coco_instances(dataset_name, {},ann_path, imgs_path)
    except:
        print(f'{dataset_name} already registered' )
    print(f'loading images from {imgs_path}')
    print(f'loading annotations from {ann_path}')

    dataset_dicts=DatasetCatalog.get(dataset_name)
    cat_name_arr=MetadataCatalog.get(dataset_name).thing_classes
    return dataset_dicts,cat_name_arr


def get_per_cat_instance_stats(cat_name_arr,dataset_dicts):  
    per_img_count_dict=Counter({i:[] for i in range(len(cat_name_arr))})

    for data in dataset_dicts:
        anns=data['annotations']
        cat_count_dict={i:0 for i in range(len(cat_name_arr))}
        for ann in anns:
            cat_id=ann['category_id']
            cat_count_dict[cat_id]+=1
        cat_count_dict={item[0]:[item[1]] for item in cat_count_dict.items()}
        per_img_count_dict+=Counter(cat_count_dict)

    per_img_count_dict={item[0]:np.array(item[1]) for item in per_img_count_dict.items()}

    n_obj_total=sum([sum(val) for val in per_img_count_dict.values() ])
    a=[]
    for key,val in per_img_count_dict.items():
    #     print(f'class: {cat_name_arr[key]}')
        obj_ratio=sum(val)/n_obj_total
    #     print(f'object ratio: {obj_ratio}')
    #     print('_________________________________')
        a.append(obj_ratio)
    assert round(sum(a))==1
    cat_prob_series=pd.Series(a,index=cat_name_arr).sort_values()
    return cat_prob_series

def get_n_instance_stats(dataset_dicts):
    n_instance_lst=[]
    for data in dataset_dicts:
        n_instance_lst.append(len(data['annotations']))
    grouped_n_intance=pd.Series(n_instance_lst).value_counts()
    n_instance_prob=grouped_n_intance/grouped_n_intance.sum()
    return n_instance_prob

def get_stats(dataset_name,source):
    dataset_dicts,cat_name_arr=register_load_dataset(dataset_name,source)
    cat_prob_series=get_per_cat_instance_stats(cat_name_arr,dataset_dicts)
    n_instance_prob=get_n_instance_stats(dataset_dicts)
    
    return cat_prob_series,n_instance_prob