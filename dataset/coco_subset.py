import os,shutil
from coco_dataset import *
import json

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

        # coco sub anno json
        self.sub_anno_file=os.path.join(os.path.dirname(self.coco_anno_file),f'{out_folder_prefix}_subset.json')

    def create_coco_subset(self):
        # 初始化 COCO API
        coco = COCO(self.coco_anno_file)

        # 获取目标类别的 ID
        category_ids = self.sub_category_id
        print(f"目标类别 ID: {category_ids}")

        # 获取包含目标类别的图像 ID
        img_ids = []
        for cat_id in category_ids:
            img_ids.extend(coco.getImgIds(catIds=cat_id))
        img_ids = list(set(img_ids))  # 去重

        # 筛选出仅包含目标类别的图像
        filtered_img_ids = []
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids)
            if len(ann_ids) > 0:
                filtered_img_ids.append(img_id)

        # 构建新的标注文件
        new_anno = {
            "images": [],
            "annotations": [],
            "categories": coco.loadCats(category_ids)
        }

        # 收集图像信息
        image_id_map = {}
        for img_id in filtered_img_ids:
            img_info = coco.loadImgs(img_id)[0]
            new_img_id = len(new_anno["images"]) + 1
            image_id_map[img_id] = new_img_id
            img_info["id"] = new_img_id
            new_anno["images"].append(img_info)

        # 收集标注信息
        anno_id = 1
        for img_id in filtered_img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            ann["id"] = anno_id
            ann["image_id"] = image_id_map[img_id]
            new_anno["annotations"].append(ann)
            anno_id += 1

        # 保存新的标注文件
        with open(self.sub_anno_file, 'w') as f:
            json.dump(new_anno, f)

        print(f"子集创建完成，包含 {len(filtered_img_ids)} 张图像和 {anno_id - 1} 个标注。")


if __name__=="__main__":
    # sub_category_id=[1,2,3] # person, bicycle, car
    sub_category_id=[3] # person, bicycle, car

    coco_sub_train=COCOSubset(sub_category_id=sub_category_id,
                              coco_img_dir=coco_train_img_dir,
                              coco_anno_file=coco_train_annotation_file,
                              out_folder_prefix='train')
    coco_sub_train.create_coco_subset()

    coco_sub_val=COCOSubset(sub_category_id=sub_category_id,
                            coco_img_dir=coco_val_img_dir,
                            coco_anno_file=coco_val_annotation_file,
                            out_folder_prefix='val')
    coco_sub_val.create_coco_subset()





