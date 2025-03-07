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

class ImgLabelResize:
    def __init__(self):
        pass

    @classmethod
    def image_resize(cls, img:Image, new_size=224)->Image:
        # Calculate the new size while maintaining aspect ratio
        width, height = img.size

        if width > height:
            new_width = new_size
            new_height = int(height * (new_size / width))
        else:
            new_height = new_size
            new_width = int(width * (new_size / height))

        # Resize the image
        resized_image = img.resize((new_width, new_height), Image.BICUBIC)

        # Create a new blank square image
        square_image = Image.new("RGB", (new_size, new_size))

        # Calculate the position to paste the resized image
        x_offset = (new_size - new_width) // 2
        y_offset = (new_size - new_height) // 2

        # Paste the resized image onto the square image
        square_image.paste(resized_image, (x_offset, y_offset))

        return square_image

    @classmethod
    def label_resize(cls, origin_width, origin_height, bbox:list[int],new_size=224)->list[int]:
        # ratio
        ratio=min(new_size/origin_width, new_size/origin_height)

        # padding_width
        padding_width_half=abs(new_size-origin_width*ratio)//2
        padding_height_half=abs(new_size-origin_height*ratio)//2
        print(f'origin_width:{origin_width},origin_height:{origin_height}')
        print(f'padding_width_half:{padding_width_half},padding_height_half:{padding_height_half}')

        # x,y,width,height
        x,y,width,height=bbox[0],bbox[1],bbox[2],bbox[3]
        x=int(ratio*x+padding_width_half)
        y=int(ratio*y+padding_height_half)
        width=int(ratio*width)
        height=int(ratio*height)

        return [x,y,width,height]

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

    def load_img(self,img_file_name)->Image:
        '''
        return PIL Image with shape (width, height, channel)
        '''
        img=Image.open(os.path.join(self.img_dir, img_file_name))
        return img

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

        # load image
        img_data=self.load_img(self.get_img_name(img_info))
        print(f'img shape:{img_data.size}')
        origin_width,origin_height=img_data.size
        img_data=ImgLabelResize.image_resize(img=img_data,new_size=224)

        # get image annotation of an image
        anno_infos=self.get_annotation_infos_by_img_id(self.get_img_id(img_info))
        for anno_info in anno_infos:
            print(f'anno_info, category_id:{anno_info["category_id"]}, bbox:{anno_info["bbox"]}')
            cat_info=self.get_category_info(anno_info["category_id"])
            print(f'category_id:{anno_info["category_id"]}, cat_info:{cat_info}')
            anno_info['bbox']=ImgLabelResize.label_resize(origin_width,origin_height,anno_info['bbox'],224)
            anno_info['segmentation']=[] # clear segmentation for now

        plt.imshow(img_data)
        self.coco.showAnns(anno_infos, draw_bbox=True)
        plt.show()


if __name__=="__main__":
    coco_train_parser=COCOParser(img_dir=coco_train_img_dir, annotation_file=coco_train_annotation_file)
    coco_train_parser.test_get_annotation()

    coco_test_parser=COCOParser(img_dir=coco_val_img_dir, annotation_file=coco_val_annotation_file)
    coco_test_parser.test_get_annotation()


