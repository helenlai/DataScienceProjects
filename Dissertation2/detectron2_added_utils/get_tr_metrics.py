import os
import numpy as np
import json
import matplotlib.pyplot as plt
from detectron2.config import get_cfg


def load_lines(file_path):
    with open(file_path,'r') as f:
        lines=f.readlines()
    return lines

def unpack_metrics(config_name):
    config_path_prefix='/redresearch1/hlai/detectron2/configs/'
    config_path=config_path_prefix+config_name
    cfg=get_cfg()
    cfg.set_new_allowed(True)

    # config_path='/redresearch1/hlai/detectron2/configs/x101_round2_v2.yaml'
    cfg.merge_from_file(config_path)
    out_dir=cfg.OUTPUT_DIR
    # tr_metrics_path=os.path.join(out_dir,'train_metrics.json')
    metrics_path=os.path.join(out_dir,'metrics.json')

    lines=load_lines(metrics_path)
    tr_metrics_names=['total_loss','fast_rcnn/cls_accuracy','lr','iteration']
    metric_dict={metric_name: [] for metric_name in tr_metrics_names}

    for line in lines:
        line_dict=json.loads(line)
        try:
            for metric_name in tr_metrics_names:
                val_lst=metric_dict[metric_name]
                val_lst.append(line_dict[metric_name])
                metric_dict[metric_name]=val_lst
        except:
            print(line)
            pass

    plt.plot(metric_dict['iteration'],metric_dict['total_loss'])
    plt.title('total training loss')
    plt.savefig(out_dir+'/tr_loss_trend.jpg')
    plt.show()
    
    with open(out_dir+'/tr_metrics_dict.json','w') as file:
        json.dump(metric_dict,file)
     
    return metric_dict
