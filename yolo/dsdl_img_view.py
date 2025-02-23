import random
from dsdl.dataset import DSDLDataset
from PIL import Image, ImageDraw
import os

g_file_path=os.path.dirname(os.path.abspath(__file__))

val_yaml = os.path.join(g_file_path,"..","dataset","OpenDataLab___COCO_2017",
                        "dsdl","dsdl_Det+InsSeg_full","set-val","val.yaml")

loc_config = dict(
    type="LocalFileReader",
    working_dir=os.path.join(g_file_path,"..","dataset","OpenDataLab___COCO_2017",
                        "raw","Images")
)

# 初始化Dataset
ds_val = DSDLDataset(dsdl_yaml=val_yaml, location_config=loc_config)

# 获取索引为0的样本
example = ds_val[0]
print(example.keys())

# 提取图片
img = example.Image[0].to_image().convert(mode='RGB')
print(f'{img}')

# 定义Draw方法
draw = ImageDraw.Draw(img)

# 迭代绘制标注框及其类别名称
for i in range(len(example.Bbox)):
    color = (random.randint(0,250), random.randint(0,250), random.randint(0,250))
    draw.rectangle(example.Bbox[i].xyxy, width=2, outline=color)
    x,y,w,h = example.Bbox[i].xywh
    draw.text((x,y), example.Label[i].name)

# 展示绘图结果
img.show()