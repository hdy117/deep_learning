from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np

file_path=os.path.dirname(os.path.abspath(__file__))

# coco root dir
coco_root_dir=os.path.join(file_path,'dataset','OpenDataLab___COCO_2017','raw')
coco_img_dir=os.path.join(coco_root_dir,'Images')
coco_annotation_dir=os.path.join(coco_root_dir,'Annotations')

# coco annotation of train
coco_train_img_dir=os.path.join(coco_img_dir,'train2017')
coco_train_annotation_file=os.path.join(coco_annotation_dir,'annotations_trainval2017','annotations','instances_train2017.json')

# coco annotation of val
coco_val_img_dir=os.path.join(coco_img_dir,'val2017')
coco_val_annotation_file=os.path.join(coco_annotation_dir,'annotations_trainval2017','annotations','instances_val2017.json')


class COCOParser:
    def __init__(self, img_dir=coco_train_img_dir, annotation_file=coco_train_annotation_file):
        '''
        {
            "images": [
                {
                    "id": 1,
                    "width": 640,
                    "height": 480,
                    "file_name": "image_001.jpg",
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "2013-11-14 17:02:52"
                },
                ...
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [
                        [
                            100, 100, 200, 100, 200, 200, 100, 200
                        ]
                    ],
                    "area": 10000,
                    "bbox": [x, y, width, height],
                    "iscrowd": 0
                },
                ...
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person"
                },
                ...
            ],
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                },
                ...
            ]
            }
        '''
        # coco parser
        self.img_dir=img_dir
        self.anno_file=annotation_file
        self.coco:COCO=COCO(annotation_file)
    
    def get_img_infos(self)->list[dict]:
        img_ids=self.coco.getImgIds()
        img_infos=self.coco.loadImgs(img_ids)
        return img_infos

    def get_img_num(self)->int:
        return len(self.get_img_infos())
    
    def get_img_name(self, img_info)->str:
        return img_info['file_name']
    
    def get_img_id(self, img_info)->int:
        return img_info['id']

    def load_img(self, img_file_name)->np.ndarray:
        '''
        return numpy array with shape (width, height, channel)
        '''
        img=Image.open(os.path.join(self.img_dir, img_file_name))
        return np.array(img)

    def get_annotation_infos_by_img_id(self, img_id)->list[dict]:
        anno_id=self.coco.getAnnIds(imgIds=img_id)
        anno_infos=self.coco.loadAnns(anno_id)
        return anno_infos

    def get_category_info(self, category_id) -> dict:
        return self.coco.loadCats(category_id)[0]
    
    def test_get_annotation(self):
        # category
        train_catgory_ids=self.coco.getCatIds()
        print(f'train_catgory_ids:{train_catgory_ids}')

        # num of images
        print(f'get_img_num:{self.get_img_num()}')

        # get image info
        img_infos=self.get_img_infos()
        img_info=img_infos[13]
        print(f'img_name:{self.get_img_name(img_info)}, img_id:{self.get_img_id(img_info)}')

        # get image annotation of an image
        anno_infos=self.get_annotation_infos_by_img_id(self.get_img_id(img_info))
        for anno_info in anno_infos:
            print(f'anno_info, category_id:{anno_info["category_id"]}, bbox:{anno_info["bbox"]}')
            cat_info=self.get_category_info(anno_info["category_id"])
            print(f'category_id:{anno_info["category_id"]}, cat_info:{cat_info}')

        img_data=self.load_img(self.get_img_name(img_info))
        print(f'img shape:{img_data.shape}')
        plt.imshow(img_data)
        self.coco.showAnns(anno_infos, draw_bbox=True)
        plt.show()


if __name__=="__main__":
    coco_train_parser=COCOParser(img_dir=coco_train_img_dir, annotation_file=coco_train_annotation_file)
    coco_train_parser.test_get_annotation()

    coco_test_parser=COCOParser(img_dir=coco_val_img_dir, annotation_file=coco_val_annotation_file)
    coco_test_parser.test_get_annotation()


