
import numpy as np
import random
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def enumerate_layouts(dataset_name,anns_path,imgs_path):
    register_coco_instances(dataset_name, {},anns_path, imgs_path)
    dataset_dict=DatasetCatalog.get(dataset_name)

    layout_dict={}
    for data in dataset_dict:
        category_lst=[]
        anns=data['annotations']
        for ann in anns:
            category_lst.append(ann['category_id'])
        category_tuple=tuple(set(category_lst))
        if category_tuple in layout_dict.keys():
            layout_dict[category_tuple]+=1
        else:
            layout_dict[category_tuple]=1
    
    ranked_tuples=sorted(layout_dict.items(),key=lambda item : item[1],reverse=True)
    return ranked_tuples



def layout_id2name(top_layouts,cat_name_arr):
    layout_lsts=[]
    layout_count=[]
    for tup in top_layouts:
        cat_id_tup=tup[0]
        layout_lst=[cat_name_arr[cat_id] for cat_id in cat_id_tup]

        if len(layout_lst)>2 and all(np.isin(['title','forms'],layout_lst)):
            continue
        else:
            layout_lsts.append(layout_lst)
            layout_count.append(tup[1])
    return layout_lsts,np.array(layout_count)
                               
    
    
def get_templates(layout_lsts):
    layouts=[]
    p_arrs=[]

    for layout in layout_lsts:

        if (len(layout)==2) and ('title' in layout):
            assert 'caption' not in layout
            main_obj=[cat for cat in layout if cat!='title']
            layout_components2=main_obj+['title',True] 
            p_arr=np.array([0.7,0.3])



        elif len(layout)>2:
            layout_components=np.array(layout)
            layout_components1=layout_components[(layout_components!='caption') & (layout_components!='title')]
            layout_components2=layout_components[layout_components!='caption']  


            main_obj=random.choice(layout_components1)
            p_main=1.5/len(layout_components2)
            p_other=(1-p_main)/(len(layout_components2)-1)

            p_arr=np.zeros(len(layout_components2))
            p_arr[layout_components2==main_obj]=p_main
            p_arr[layout_components2!=main_obj]=p_other

        else:
            layout_components=np.array(layout)
            layout_components2=layout_components[layout_components!='caption']  
            if len(layout_components2)!=1:
                print('problem encountered')
                print(layout)
                break
            p_arr=np.array([1])

        layouts.append(layout_components2)
        p_arrs.append(p_arr)

    return layouts,p_arrs
        
def create_templates(dataset_name,anns_path,imgs_path,top_n): 

    ranked_tuples=enumerate_layouts(dataset_name,anns_path,imgs_path)
    cat_name_arr=MetadataCatalog.get(dataset_name).thing_classes
    top_layouts=ranked_tuples[:top_n]
    layout_lsts,layout_count=layout_id2name(top_layouts,cat_name_arr)
    # layout_prob=layout_count/layout_count.sum()
    layouts,p_arrs=get_templates(layout_lsts)
    
    
    return layouts,p_arrs,layout_count
