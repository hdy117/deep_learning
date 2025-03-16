import torch
from PIL import Image
import os
import torchvision.transforms as transforms

def apply_cdf_to_channels(image_tensor):
    # 确保输入是3通道图像
    assert image_tensor.shape[0] == 3, "Input must be a 3-channel tensor"
    
    processed_channels = []
    for c in range(3):
        channel = image_tensor[c]
        flat_channel = channel.view(-1)
        
        # 计算直方图（假设像素值在0-1之间）
        bins = 256
        min_val, max_val = flat_channel.min(), flat_channel.max()
        if max_val == min_val:
            # 处理全同值通道（避免除以零）
            processed_channels.append(channel/255.0)
            continue
        
        hist = torch.histc(flat_channel, bins=bins, min=min_val, max=max_val)
        pdf = hist / hist.sum()
        cdf = pdf.cumsum(dim=0)
        
        # 归一化CDF到0-1范围
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        
        # 创建映射表
        bin_edges = torch.linspace(min_val, max_val, bins + 1)
        
        # 将像素值映射到CDF
        indices = torch.bucketize(flat_channel, bin_edges) - 1  # 转换为0-based索引
        indices = torch.clamp(indices, 0, bins - 1)
        new_flat = cdf[indices]
        
        # 恢复通道形状
        processed_channels.append(new_flat.view(channel.shape))
    
    return torch.stack(processed_channels, dim=0)

# 示例使用
if __name__ == "__main__":
    img_file=os.path.join('OpenDataLab___COCO_2017','sample','000000000009.jpg')
    img_pil=Image.open(img_file).convert("RGB")
    
    # pil to tensor
    pil_to_tensor=transforms.ToTensor()
    img_tensor=pil_to_tensor(img_pil)

    # apply cdf 
    img_tensor_cdf=apply_cdf_to_channels(img_tensor)

    # tensor to pip img
    tensor_to_pil=transforms.ToPILImage()
    img_pil=tensor_to_pil(img_tensor_cdf)

    # show
    img_pil.show()