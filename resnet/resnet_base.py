import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,sys,time
import matplotlib.pyplot as plt

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(g_file_path,".."))

from dataset import coco_dataset

# coco dataset
class COCODataset(Dataset):
    def __init__(self, img_folder, anno_file_path, img_new_size=224, target_class:list=coco_dataset.coco_10_categories, transform=None, augmentation=True):
        super().__init__()
        self.transform=transform
        self.target_class=target_class
        self.img_new_size=img_new_size
        self.coco_parser=coco_dataset.COCOParser(img_dir=img_folder, annotation_file=anno_file_path, execlusive_category_id=1,augmentation=augmentation)
        self.coco_parser.set_target_category_id(target_category_ids=target_class)
        self.img_infos=self.coco_parser.get_img_infos()

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        """
        # label, [num_class]
        # image, [channel,width,height], range [0,255]
        """
        img_info=self.img_infos[index]
        img_id=self.coco_parser.get_img_id(img_info)

        # load and resize image
        img_pil=self.coco_parser.load_img(self.coco_parser.get_img_name(img_info=img_info))
        # origin_width, origin_height=img_pil.size
        img_pil=coco_dataset.ImgLabelResize.image_resize(img=img_pil,new_size=self.img_new_size)
        # new_width, new_height=img_pil.size

        # labels
        labels=torch.zeros(max(self.target_class))

        # load and resize labels
        anno_infos=self.coco_parser.get_annotation_infos_by_img_id(img_id)

        for anno_info in anno_infos:
            # get category id
            category_id=anno_info['category_id']
            
            # only detect category in HyperParam.TARGET_CLASS_LABELS
            if category_id in self.target_class:
                # num class
                labels[category_id-1]=1.0 # class

        # show img
        # print(f'labels:{labels}')
        # img_pil.show()

        if self.transform:
            img_pil=self.transform(img_pil)
            # img_pil=image_cdf.apply_cdf_to_channels(img_pil)
        
        # print img
        # print(f'img_pil:{img_pil.mean()}')

        return img_pil, labels

# train dataloader
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.RandomRotation(180),
    transforms.ToTensor()
])

# hyper param
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path=os.path.join(g_file_path,"resnet18.pth")
batch_size=128
n_epoch=40
lr=0.0001
weight_decay=0.0005
lr_step_size=n_epoch//2
img_new_size=224
target_class=coco_dataset.coco_10_categories # coco category [bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light]
out_dim=max(target_class)
# target_class=[val for val in range(1,91)] # coco category [person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light]

if __name__=="__main__":
    train_dataset=COCODataset(coco_dataset.coco_train_img_dir,
                          coco_dataset.coco_train_annotation_file,
                          img_new_size=img_new_size,
                          target_class=target_class,
                          transform=transform)
    sample,label=train_dataset[12]
    print(f'{sample.mean()}')