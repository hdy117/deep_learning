from PIL import Image, ImageDraw
import os
import shutil
import torch
import coco_dataset
import matplotlib.pyplot as plt

g_file_path=os.path.dirname(os.path.abspath(__file__))

class BBOXUtils:
    def __init__(self, out_folder:str=g_file_path):
        self.img_idx:int=0
        self.img_prefix:str="img_"
        self.out_folder:str=out_folder

        os.makedirs(out_folder, exist_ok=True)
    
    def save_one_img_with_bbox(self,img:Image, bboxs:list[list[int]],img_idx:int=0):
        '''
        draw resized image and labels
        '''
        img_pil:Image = img
        # print(f"img center pixel:{np.array(img_pil)[112,112]}")

        width,height=img_pil.size
        draw = ImageDraw.Draw(img_pil)
        
        # get and draw bbox
        for bbox in bboxs:
            # extract bbox if condidence is greater than conf_thresh
            [x,y,w,h]=bbox
            x_top_left=int(x-w/2)
            y_top_left=int(y-h/2)
            x_down_right=int(x+w/2)
            y_down_right=int(y+h/2)
            print(f'{x_top_left},{y_top_left},{x_down_right},{y_down_right}')

            # draw bbox
            draw.rectangle([x_top_left,y_top_left,x_down_right,y_down_right], outline="red", width=2)  # outline颜色，width线宽
        
        # save image
        out_img_path=os.path.join(self.out_folder, f'{self.img_prefix}{img_idx}.jpg')
        img_pil.save(out_img_path)

    def save_one_norm_img_with_bbox(self, imgs:torch.Tensor, pred_class:torch.Tensor, 
                           pred_conf:torch.Tensor, pred_bbox:torch.Tensor, 
                           grid_size:int=224//7, conf_thresh=0.6, img_idx:int=0):
        '''
        save resized and image from yolo v1
        '''
        # convert img tensor to pil image
        img_tensor:torch.Tensor=imgs.cpu() # img tensor
        img_tensor=img_tensor.permute(1,2,0)
        img_tensor = (img_tensor*255).clip(0,255).byte()
        img_pil:Image = Image.fromarray(img_tensor.numpy(),mode="RGB")
        # print(f"img center pixel:{np.array(img_pil)[112,112]}")

        width,height=img_pil.size
        draw = ImageDraw.Draw(img_pil)
        
        # get and draw bbox
        for grid_i in range(pred_bbox.shape[0]):
            for grid_j in range(pred_bbox.shape[1]):
                if pred_conf[grid_i,grid_j]>conf_thresh:
                    # extract bbox if condidence is greater than conf_thresh
                    [x,y,w,h]=pred_bbox[grid_i, grid_j].tolist()

                    # de-normalize bbox
                    x=grid_i*grid_size+x*grid_size
                    y=grid_j*grid_size+y*grid_size
                    if w<=0.0 or h<=0.0:
                        continue
                    print(f'{x},{y},{w},{h}')
                    w=width*w
                    h=height*h
                    x_top_left=int(x-w/2)
                    y_top_left=int(y-h/2)
                    x_down_right=int(x+w/2)
                    y_down_right=int(y+h/2)
                    print(f'{x_top_left},{y_top_left},{x_down_right},{y_down_right}')

                    # draw bbox
                    draw.rectangle([x_top_left,y_top_left,x_down_right,y_down_right], outline="red", width=2)  # outline颜色，width线宽
        
        # save image
        out_img_path=os.path.join(self.out_folder, f'{self.img_prefix}{img_idx}.jpg')
        img_pil.save(out_img_path)
    
    def save_norm_imgs_with_bbox(self, imgs:torch.Tensor, pred_class:torch.Tensor, 
                           pred_conf:torch.Tensor, pred_bbox:torch.Tensor, 
                           grid_size:int=224//7, conf_thresh=0.6):
        '''
        save image with bbox for YOLO v1 only
        '''
        print(imgs.shape)
        batch_size=imgs.shape[0] # batch size

        for i in range(batch_size):
            # convert img tensor to pil image
            img_tensor:torch.Tensor=imgs[i].cpu() # img tensor
            img_tensor=img_tensor.permute(1,2,0)
            img_tensor = (img_tensor*255).clip(0,255).byte()
            img_pil:Image = Image.fromarray(img_tensor.numpy(),mode="RGB")
            # print(f"img center pixel:{np.array(img_pil)[112,112]}")

            width,height=img_pil.size
            draw = ImageDraw.Draw(img_pil)
            
            # get and draw bbox
            for grid_i in range(pred_bbox.shape[1]):
                for grid_j in range(pred_bbox.shape[2]):
                    if pred_conf[i,grid_i,grid_j]>conf_thresh:
                        # extract bbox if condidence is greater than conf_thresh
                        [x,y,w,h]=pred_bbox[i, grid_i, grid_j].tolist()

                        # de-normalize bbox
                        x=grid_i*grid_size+x*grid_size
                        y=grid_j*grid_size+y*grid_size
                        if w<=0.0 or h<=0.0:
                            continue
                        print(f'{x},{y},{w},{h}')
                        w=width*w
                        h=height*h
                        x_top_left=int(x-w/2)
                        y_top_left=int(y-h/2)
                        x_down_right=int(x+w/2)
                        y_down_right=int(y+h/2)
                        print(f'{x_top_left},{y_top_left},{x_down_right},{y_down_right}')

                        # draw bbox
                        draw.rectangle([x_top_left,y_top_left,x_down_right,y_down_right], outline="red", width=2)  # outline颜色，width线宽
            
            # save image
            out_img_path=os.path.join(self.out_folder, f'{self.img_prefix}{self.img_idx}.jpg')
            img_pil.save(out_img_path)
            self.img_idx+=1

        
if __name__=="__main__":
    coco_parser=coco_dataset.COCOParser(img_dir=coco_dataset.coco_train_img_dir, annotation_file=coco_dataset.coco_train_sub_annotation_file)
    bbox_utils=BBOXUtils(out_folder="bbox_test")

    # num of images
    print(f'get_img_num:{coco_parser.get_img_num()}')

    # get image info
    img_infos=coco_parser.get_img_infos()
    img_idx=0
    for img_info in img_infos:
        print(f'img_name:{coco_parser.get_img_name(img_info)}, img_id:{coco_parser.get_img_id(img_info)}')

        # load image
        img_data=coco_parser.load_img(coco_parser.get_img_name(img_info))
        print(f'img shape:{img_data.size}')
        origin_width,origin_height=img_data.size
        img_data=coco_dataset.ImgLabelResize.image_resize(img=img_data,new_size=224)

        # get image annotation of an image
        anno_infos=coco_parser.get_annotation_infos_by_img_id(coco_parser.get_img_id(img_info))
        bboxs=[]
        for anno_info in anno_infos:
            print(f'anno_info, category_id:{anno_info["category_id"]}, bbox:{anno_info["bbox"]}')
            cat_info=coco_parser.get_category_info(anno_info["category_id"])
            print(f'category_id:{anno_info["category_id"]}, cat_info:{cat_info}')
            anno_info['bbox']=coco_dataset.ImgLabelResize.label_resize(origin_width,origin_height,anno_info['bbox'],224)
            anno_info['segmentation']=[] # clear segmentation for now
            print(f'bbox:{anno_info["bbox"]}')
            bboxs.append(anno_info["bbox"])
        
        # save resized img and label
        bbox_utils.save_one_img_with_bbox(img=img_data,bboxs=bboxs,img_idx=img_idx)
        img_idx+=1
    

