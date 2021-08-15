import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

def load_json(json_path):
    with open(json_path,'r') as f:
        json_file=json.load(f)
    return json_file


def get_out_dir(config_file,folder_prefix='/redresearch1/hlai/detectron2/'):
    temp=config_file.split('/')[-1]
    out_folder=temp.split('.')[0]
    out_dir=os.path.join(folder_prefix,out_folder)
    return out_dir

def save_lst(lst,fname,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    f_path=os.path.join(out_dir,fname)
    open(f_path, 'a').close()
    with open(f_path,'wb') as f:
        pickle.dump(lst, f)

def plot_val_AP(config_lst,out_dir_):
    
    AP_lst=[]
    iter_lst=[]
    

    for i,config_path in enumerate(config_lst):
        out_dir=get_out_dir(config_path)
        result_path=out_dir+'/val_result_dict.json'
        val_result_dict=load_json(result_path)
        APs=val_result_dict['bbox/AP']
        AP_lst+=APs
        iters=val_result_dict['iteration']
        if i==0:
            round_len=iters[-1]

        else:
            iters=[iter_+round_len*i for iter_ in iters]
        iter_lst+=iters
    assert len(AP_lst)==len(iter_lst)

    save_lst(AP_lst,'AP_list.txt',out_dir_)
    save_lst(iter_lst,'iter_list.txt',out_dir_)


    os.makedirs(out_dir_,exist_ok=True)
    plt.plot(iter_lst,AP_lst)
    plt.title('validation AP during training')
    plt.savefig(os.path.join(out_dir_,'val_AP_trend.jpg'))
    plt.show()

    