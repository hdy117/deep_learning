import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import os
import torchvision.transforms as transforms

# file_path
g_file_path=os.path.dirname(os.path.abspath(__file__))
coco_train_dir=os.path.join(g_file_path, "..","dataset", "OpenDataLab___COCO_2017", 'raw')

class COCODatasetResized(Dataset):
    def __init__(self, root, annotation, img_size=224,num_classes=80):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.img_size = img_size  # Target size: 448x448
        self.num_classes=num_classes
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),  
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        annotation_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(annotation_ids)
        image_info = self.coco.loadImgs(img_id)[0]

        image_path = os.path.join(self.root, image_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found!")
            return self.__getitem__((index + 1) % len(self.ids))  # Skip to next image

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = self.transform(image)  # Apply torchvision transforms

        # Adjust bounding boxes
        label_matrix = self.encode_label(annotations, orig_w, orig_h)

        return image, label_matrix

    def encode_label(self, annotations, orig_w, orig_h):
        label_matrix = torch.zeros((7, 7, 5*2 + self.num_classes))  # Supports COCO (5 + 80 = 85)

        # Get valid category IDs (COCO category mapping)
        cat_ids = self.coco.getCatIds()  # Returns a sorted list of valid category IDs
        cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

        for ann in annotations:
            x, y, w, h = ann['bbox']
            class_label = ann['category_id']

            # Ensure class label is valid
            if class_label not in cat_id_to_index:
                continue  # Skip unknown categories

            class_label = cat_id_to_index[class_label]  # Convert to 0-based index

            # Scale bounding box coordinates
            x_new = (x / orig_w) * self.img_size
            y_new = (y / orig_h) * self.img_size
            w_new = (w / orig_w) * self.img_size
            h_new = (h / orig_h) * self.img_size

            # Compute grid cell indices
            grid_size = self.img_size / 7
            grid_x = int(x_new // grid_size)
            grid_y = int(y_new // grid_size)

            # Normalize x, y within the grid cell
            x_rel = (x_new % grid_size) / grid_size
            y_rel = (y_new % grid_size) / grid_size
            w_rel = w_new / self.img_size
            h_rel = h_new / self.img_size

            # Assign values to the label matrix
            label_matrix[grid_y, grid_x, 0:4] = torch.tensor([x_rel, y_rel, w_rel, h_rel])
            label_matrix[grid_y, grid_x, 4] = 1  # Objectness score
            label_matrix[grid_y, grid_x, 5 + class_label] = 1  # One-hot class label

        return label_matrix

    def __len__(self):
        return len(self.ids)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = COCODatasetResized(root=os.path.join(coco_train_dir,"Images",'train2017'), 
                                   annotation=os.path.join(coco_train_dir,'Annotations','annotations_trainval2017','annotations','instances_train2017.json'))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # Calculate coordinate loss (bounding box)
        coord_loss = self.lambda_coord * torch.sum((predictions[..., 0:4] - targets[..., 0:4]) ** 2)
        
        # Calculate object loss (objectness score)
        object_loss = torch.sum((predictions[..., 4] - targets[..., 4]) ** 2)
        
        # Calculate no-object loss (for empty grid cells)
        no_object_loss = self.lambda_noobj * torch.sum((predictions[..., 4] - targets[..., 4]) ** 2)
        
        # Calculate class loss (class prediction)
        class_loss = torch.sum((predictions[..., 5:] - targets[..., 5:]) ** 2)
        
        # Combine all losses
        total_loss = coord_loss + object_loss + no_object_loss + class_loss
        return total_loss



class YOLO(nn.Module):
    def __init__(self, num_classes=80, grid_size=7, num_boxes=2):
        super(YOLO, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * grid_size * grid_size, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, grid_size * grid_size * (num_boxes * 5 + num_classes))
        )
        
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes

    def forward(self, x):
        x = self.conv_layers(x)
        # print("Conv output shape:", x.shape)  # Debugging
        x = self.fc_layers(x)
        return x.view(-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = YoloLoss()

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx ,(images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()


def yolo_to_bbox(yolo_output, img_size=448, grid_size=7, num_boxes=2, num_classes=80):
    """
    Convert YOLO output (grid-based format) to bounding box coordinates in the original image scale.
    """
    batch_size = yolo_output.shape[0]
    bboxes = []
    
    cell_size = img_size / grid_size

    for batch in range(batch_size):
        img_boxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell_data = yolo_output[batch, i, j]
                
                for b in range(num_boxes):
                    box_offset = b * 5
                    confidence = cell_data[box_offset + 4]
                    if confidence < 0.5:  # Filter weak detections
                        continue

                    x_rel, y_rel, w_rel, h_rel = cell_data[box_offset: box_offset + 4]
                    x = (j + x_rel) * cell_size
                    y = (i + y_rel) * cell_size
                    w = w_rel * img_size
                    h = h_rel * img_size

                    class_scores = cell_data[num_boxes * 5:]
                    class_id = torch.argmax(class_scores).item()
                    class_prob = class_scores[class_id]

                    img_boxes.append([x, y, w, h, confidence.item(), class_id, class_prob.item()])

        bboxes.append(img_boxes)
    
    return bboxes

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Box format: [x_center, y_center, width, height]
    """
    box1 = torch.tensor([box1[0] - box1[2] / 2, box1[1] - box1[3] / 2,
                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2])
    box2 = torch.tensor([box2[0] - box2[2] / 2, box2[1] - box2[3] / 2,
                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2])
    
    return box_iou(box1.unsqueeze(0), box2.unsqueeze(0)).item()

def compute_map(model, dataloader, iou_threshold=0.5, num_classes=80):
    """
    Evaluate the model using mean Average Precision (mAP).
    """
    model.eval()
    all_detections = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)  # Get model predictions
            bboxes_pred = yolo_to_bbox(predictions)  # Convert YOLO format to bbox format
            
            for batch_idx in range(len(targets)):
                gt_bboxes = []
                for i in range(7):
                    for j in range(7):
                        if targets[batch_idx, i, j, 4] == 1:  # Object present
                            x, y, w, h = targets[batch_idx, i, j, :4].cpu().numpy()
                            class_id = torch.argmax(targets[batch_idx, i, j, 5:]).item()
                            gt_bboxes.append([x, y, w, h, class_id])

                all_detections.append(bboxes_pred[batch_idx])
                all_ground_truths.append(gt_bboxes)

    # Compute AP for each class
    average_precisions = []
    for class_id in range(num_classes):
        true_positives = []
        scores = []
        num_gt = 0

        for pred_boxes, gt_boxes in zip(all_detections, all_ground_truths):
            gt_class_boxes = [box for box in gt_boxes if box[4] == class_id]
            num_gt += len(gt_class_boxes)

            if len(gt_class_boxes) == 0:
                continue

            for pred in pred_boxes:
                if pred[5] != class_id:
                    continue
                
                scores.append(pred[4])
                best_iou = max([compute_iou(pred, gt) for gt in gt_class_boxes]) if gt_class_boxes else 0

                if best_iou > iou_threshold:
                    true_positives.append(1)
                else:
                    true_positives.append(0)

        if num_gt == 0:
            continue

        # Sort by confidence score
        sorted_indices = np.argsort(scores)[::-1]
        true_positives = np.array(true_positives)[sorted_indices]

        # Compute precision-recall curve
        cumulative_tp = np.cumsum(true_positives)
        precision = cumulative_tp / (np.arange(len(true_positives)) + 1)
        recall = cumulative_tp / num_gt

        # Compute Average Precision (AP)
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        average_precisions.append(ap)

    # Compute mean Average Precision (mAP)
    mAP = np.mean(average_precisions) if average_precisions else 0
    return mAP

# Load test dataset
test_dataset = COCODatasetResized(root=os.path.join(coco_train_dir,"Images",'test2017'), 
                                   annotation=os.path.join(coco_train_dir,'Annotations','annotations_trainval2017','annotations','instances_test2017.json'))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Compute mAP
mAP = compute_map(model, test_loader)
print(f"Mean Average Precision (mAP): {mAP:.4f}")

