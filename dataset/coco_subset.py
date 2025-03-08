import os,shutil
from coco_dataset import *

file_path=os.path.dirname(os.path.abspath(__file__))

class COCOSubset:
    # create a sub set of coco dataset by category id
    def __init__(self, sub_category_id:list[int]=[1,2,3], 
                 coco_img_dir:str=coco_train_img_dir, 
                 coco_anno_file:str=coco_train_annotation_file,
                 out_folder_prefix:str='train'):
        self.sub_category_id:list[int]=sub_category_id
        self.coco_img_dir:str=coco_img_dir
        self.coco_anno_file:str=coco_anno_file
        
        # coco parser
        self.coco_parser:COCOParser=COCOParser(img_dir=self.coco_img_dir,annotation_file=self.coco_anno_file)

        # coco sub image dir
        self.sub_img_dir=os.path.join(self.coco_img_dir,'..',f'{out_folder_prefix}_subset')

        # create dir if not exist
        os.makedirs(name=self.sub_img_dir,exist_ok=True)

    def create_sub_img_set(self):
        img_infos=self.coco_parser.get_img_infos()
        for img_info in img_infos:
            img_id=self.coco_parser.get_img_id(img_info=img_info)
            anno_infos=self.coco_parser.get_annotation_infos_by_img_id(img_id=img_id)
            for anno_info in anno_infos:
                if anno_info['category_id'] in self.sub_category_id:
                    img_name=self.coco_parser.get_img_name(img_info=img_info)
                    img_src_full_path=os.path.join(self.coco_img_dir,img_name)
                    img_des_full_path=os.path.join(self.sub_img_dir,img_name)
                    shutil.copy(img_src_full_path,img_des_full_path)

if __name__=="__main__":
    sub_category_id=[1,2,3] # person, bicycle, car

    coco_sub_train=COCOSubset(sub_category_id=sub_category_id,
                              coco_img_dir=coco_train_img_dir,
                              coco_anno_file=coco_train_annotation_file,
                              out_folder_prefix='train')
    coco_sub_train.create_sub_img_set()

    coco_sub_val=COCOSubset(sub_category_id=sub_category_id,
                            coco_img_dir=coco_val_img_dir,
                            coco_anno_file=coco_val_annotation_file,
                            out_folder_prefix='val')
    coco_sub_val.create_sub_img_set()





