#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import numpy as np
import json
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.added_utils.check_stp import *
from detectron2.added_utils.get_tr_metrics import *
from detectron2.added_utils.vis_val_result import *

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res



def register_dataset(data_dir,dataset_name):
    anns_path=os.path.join(data_dir,'annotations','instances.json')
    imgs_path=os.path.join(data_dir,'images')
    register_coco_instances(dataset_name, {},anns_path, imgs_path)
    print(f'{dataset_name} registered with the data dir : {data_dir}')


    # # register_coco_instances("geolay_train_synth", {},"/redresearch1/hlai/geolay/tmp/mixed_v5/img_2000_v6/instances.json", "/redresearch1/hlai/geolay/tmp/mixed_v5/img_2000_v6/images")
    # # register_coco_instances("geolay_train_synth_v2", {},"/redresearch1/hlai/geolay/train_v2/annotations/instances.json", "/redresearch1/hlai/geolay/train_v2/images")
    # register_coco_instances("geolay_train", {},"/redresearch1/hlai/geolay/train_v2/annotations/instances_v2.json", "/redresearch1/hlai/geolay/train_v2/images")
    # # register_coco_instances("geolay_train_synth_v3", {},"/redresearch1/hlai/geolay/tmp/mixed_v5/img_500_geolay_syth_v3/instances.json", "/redresearch1/hlai/geolay/tmp/mixed_v5/img_500_geolay_syth_v3/images")
    
    # register_coco_instances("geolay_train_synth_v4", {},"/redresearch1/hlai/geolay/tmp/mixed_v5/img_500_geolay_syth_v4/instances.json", "/redresearch1/hlai/geolay/tmp/mixed_v5/img_500_geolay_syth_v4/images")

    # # register_coco_instances("geolay_train_synth_v3", {},"/redresearch1/hlai/geolay/tmp/mixed_v5/img_500_geolay_syth_v2/instances.json", "/redresearch1/hlai/geolay/tmp/mixed_v5/img_500_geolay_syth_v2/images")

    # print('train registered')
    # # register_coco_instances("geolay_train_v2",{},'/redresearch1/hlai/geolay/tmp/mixed_v3/instances.json','/redresearch1/hlai/geolay/tmp/mixed_v3/images')
    # # register_coco_instances("geolay_val_synth", {},"/redresearch1/hlai/geolay/tmp/mixed_v5/img_200_v3/instances.json", "/redresearch1/hlai/geolay/tmp/mixed_v5/img_200_v3/images")
    # register_coco_instances("geolay_val_synth_v2", {},"/redresearch1/hlai/geolay/val_v2/annotations/instances_v4.json", "/redresearch1/hlai/geolay/val_v2/images")

    # #register_coco_instances("geolay_val_synth", {},"/redresearch1/hlai/geolay/synthetic_val/instances.json", "/redresearch1/hlai/geolay/synthetic_val/images")
    # print('val registered')
    # # MetadataCatalog.get("geolay_train_synth_v2").set(thing_classes=['text', 'title', 'list', 'table', 'figure', 'forms', 'toc','caption'])
    # # MetadataCatalog.get("geolay_val_synth_v2").set(thing_classes=['text', 'title', 'list', 'table', 'figure', 'forms', 'toc','caption'])
    # #MetadataCatalog.get("geolay_val_v1").set(thing_classes=['text','title','list','table','figure','forms'])
    # print('classes registered!!')

def register_datasets(dataset_paths,dataset_names):
    for dataset_path,dataset_name in zip(dataset_paths,dataset_names):
        register_dataset(dataset_path,dataset_name)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # register_dataset()
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)

    tr_dataset_paths=cfg.DATASETS.TRAIN_PATH
    tr_dataset_names=cfg.DATASETS.TRAIN
    val_dataset_paths=cfg.DATASETS.TEST_PATH
    val_dataset_names=cfg.DATASETS.TEST

    register_datasets(tr_dataset_paths,tr_dataset_names)
    if val_dataset_names!=tr_dataset_names:
        register_datasets(val_dataset_paths,val_dataset_names)
    
   


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    assert os.path.exists(cfg.OUTPUT_DIR)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    print('setting up configs')
    cfg = setup(args)
    print('configs loaded')

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    print('building model from configs')
    trainer = Trainer(cfg)
    print('model built')
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

    




def get_best_record(val_best_AP_lst,config_lst,aux_outdir,round,save=True):
    record_dict={}
    # if len(val_best_AP_lst)==0:
    #     val_best_AP_lst=[0]
    #     config_lst=['placeholder.yaml']
    val_best_AP_arr=np.array(val_best_AP_lst)
    val_best_AP_arr[val_best_AP_arr!=val_best_AP_arr]=0

    curr_best_AP=max(val_best_AP_lst)
    record_dict['curr_best_AP']=curr_best_AP
    curr_best_config=config_lst[np.argmax(val_best_AP_lst)]
    record_dict['curr_best_config']=curr_best_config
    fname='val_record_'+str(round)+'.json'

    os.makedirs(aux_outdir,exist_ok=True)
    save_json(aux_outdir,fname,record_dict)

    return curr_best_AP,curr_best_config

def start_training(args,config_lst,):

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    config_lst.append(args.config_file)
    return config_lst


def run_train_loop(args,patience,slow_down_factor,config_lst,val_best_AP_lst,round):
    # setup_logger()

    logger = logging.getLogger("detectron2.trainer")
       
    # print("Command Line Args:", args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3" 
 
    print(f'device count {torch.cuda.device_count()}')
    # if config_file is not None:
    #     args.config_file=config_file
    
    aux_outdir=args.aux_outdir
    counter=0
    print(args.eval_only)
    if args.eval_only==True:
        patience=0

    while counter<=patience:
        config_lst=start_training(args,config_lst)
        #check stopping point 
        config_name= args.config_file.split('/')[-1]
        if args.eval_only==False:
            tr_metrics_dict=unpack_metrics(config_name)
            avg_tr_loss=np.mean(tr_metrics_dict['total_loss'])
            print(f'__________training round : {counter}, avg training loss : {avg_tr_loss}, config file : {config_name}__________')

            print('____________________checking stopping point________________________')
     
            stop,new_config_name,_,best_AP=check_stopping_point(config_name,slow_down_factor)
            #collecting the best validation AP from each training round
            val_best_AP_lst.append(best_AP)
            # counter+=1
            if stop:
                print('______additional patience consumed____')
                counter+=1
            else:
                print('_______no patience consumed______')
            #update config file 
            assert os.path.isfile(new_config_name)
            args.config_file=new_config_name
            print(f'______________________training to be continued with new config : {new_config_name}______________________')
            plot_val_AP(config_lst,aux_outdir)
        else:
            pass
            counter+=1
            
    return args.config_file


def summarise_run(val_best_AP_lst,config_lst,aux_outdir,round):
    curr_best_AP,curr_best_config=get_best_record(val_best_AP_lst,config_lst,aux_outdir,round)
    print(f'round {round} finished with best AP : {curr_best_AP} with config file: {curr_best_config}')
    return curr_best_AP,curr_best_config


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    eval_only=args.eval_only
    patience,slow_down_factor=args.patience,args.slow_down_factor
    switch,checkpoint_period=args.switch_dataset,args.checkpoint_period


    if switch:
        print('switching needed')
    else:
        print('no switching')
    aux_outdir=args.aux_outdir
    config_lst=[]
    val_best_AP_lst=[]
    
    round=1
    print('start running')
    print(f'slow down factor:{slow_down_factor}')
    current_cfg_file=run_train_loop(args,patience,slow_down_factor,config_lst,val_best_AP_lst,round)
    _,curr_best_config=summarise_run(val_best_AP_lst,config_lst,aux_outdir,round)
        
    cfg=get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(curr_best_config)
    best_weight_path=os.path.join(cfg.OUTPUT_DIR,'best_model.pth')
    print('___________config loaded_________')
    
   #switching training mode 
    if eval_only ==False:
        finetune=True
        patience=1
        if finetune:
            if switch:
                print('switching dataset')
                cfg.DATASETS.TRAIN=('geolay_train',)
            else:
                print('no switching')
                print(f'{cfg.DATASETS.TRAIN}')
            cfg.SOLVER.MAX_ITER=500
            cfg.SOLVER.CHECKPOINT_PERIOD=checkpoint_period
            cfg.TEST.EVAL_PERIOD=checkpoint_period
            cfg.MODEL.WEIGHTS=best_weight_path

            current_cfg_file=args.config_file

            temp_name=current_cfg_file.split('/')
            temp_name_cfg=temp_name[-1].split('.')[0]+'_ft_v1.yaml'
            temp_name=temp_name[:-1]
            temp_name.append(temp_name_cfg)

            new_config_name="/".join(temp_name)

            temp_out_dir=cfg.OUTPUT_DIR.split('/')[:-1]
            temp_out_dir.append(temp_name_cfg[:-5])
            cfg.OUTPUT_DIR="/".join(temp_out_dir)


            print(f'new config: {new_config_name}')
            print(cfg.OUTPUT_DIR)
            save_config(new_config_name,cfg)
            print(f'new config: {new_config_name} created!' )

            args.config_file=new_config_name
            slow_down_factor=1
            round+=1
            current_cfg_file=run_train_loop(args,patience,slow_down_factor,config_lst,val_best_AP_lst,round)
            _,curr_best_config=summarise_run(val_best_AP_lst,config_lst,aux_outdir,round)
            
            save_lst(config_lst,'config_lst.txt',aux_outdir)
            # summarise_run(val_best_AP_lst,config_lst,aux_outdir,round)
                
   


