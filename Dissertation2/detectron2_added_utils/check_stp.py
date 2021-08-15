import os
import numpy as np
import json
import matplotlib.pyplot as plt
from detectron2.config import get_cfg

def load_lines(file_path):
    with open(file_path,'r') as f:
        lines=f.readlines()
    return lines



def unpack_lines(lines,save=True):
    result_dict={k:[] for k in json.loads(lines[-1]).keys()}
    for line in lines:

        line_dict=json.loads(line)
        keys=line_dict.keys()

        if "bbox/AP" in list(keys):
            for key,val in line_dict.items(): 
                try:
                    result_val=result_dict[key]
                    result_val.append(val)
                    result_dict[key]=result_val
                except:
                    pass

    return result_dict

def save_config(config_name,cfg):
    with open(config_name,'w') as file:
        file.write(cfg.dump())

def save_json(out_dir,fname,data):
    with open(os.path.join(out_dir,fname),'w') as f:
        json.dump(data,f)



def check_stopping_point(config_name,slow_down_factor):
    config_path_prefix='/redresearch1/hlai/detectron2/configs/'
    config_path=config_path_prefix+config_name
    cfg=get_cfg()
    # config_path='/redresearch1/hlai/detectron2/configs/x101_round2_v2.yaml'
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    out_dir=cfg.OUTPUT_DIR
    # tr_metrics_path=os.path.join(out_dir,'train_metrics.json')
    metrics_path=os.path.join(out_dir,'metrics.json')

    lines=load_lines(metrics_path)
    result_dict=unpack_lines(lines)
    save_json(out_dir,"val_result_dict.json",result_dict)
    print('___________validation results saved__________')

    fname_lst=os.listdir(out_dir)
    model_path_lst=[]
    for fname in fname_lst:
        if 'pth' in fname.split('.'):
            model_path_lst.append(fname)

    best_idx=np.argmax(result_dict['bbox/AP'])
    best_AP=max(result_dict['bbox/AP'])
    print(f'_______________current best AP : {best_AP}________________________')
    best_model_file=model_path_lst[best_idx]
    last_idx=len(result_dict['bbox/AP'])-1
    
    weights_path="/".join(cfg.OUTPUT_DIR.split("/")[:-1])
    weights_path=os.path.join(cfg.OUTPUT_DIR,best_model_file)
    renamed_weights_path=os.path.join(out_dir,'best_model.pth')
    os.rename(weights_path,renamed_weights_path)

    cfg.MODEL.WEIGHTS=renamed_weights_path
    #update output directory
    cfg.OUTPUT_DIR=cfg.OUTPUT_DIR[:-1]+str(int(cfg.OUTPUT_DIR[-1])+1)
    new_config_name=os.path.join(config_path_prefix,cfg.OUTPUT_DIR.split('/')[-1]+'.yaml')

    cfg.SOLVER.MAX_ITER=max(int(cfg.SOLVER.MAX_ITER*slow_down_factor),50)
    cfg.SOLVER.CHECKPOINT_PERIOD=max(int(cfg.SOLVER.CHECKPOINT_PERIOD*slow_down_factor),5)
    cfg.TEST.EVAL_PERIOD=max(int(cfg.TEST.EVAL_PERIOD*slow_down_factor),5)

  
    # cfg.SOLVER.MAX_ITER=max(cfg.SOLVER.MAX_ITER,50)

    save_config(new_config_name,cfg)


    if best_idx==last_idx or best_idx==last_idx-1 or best_idx==last_idx-2:
    # if best_idx==last_idx:
        print('continue training')
        stop_bool=False
        
    
    else:  
        # print('additional patience consumed')
        # stop_iter=result_dict['iteration'][best_idx]
        stop_bool=True
        
        
    return stop_bool,new_config_name,result_dict,best_AP


