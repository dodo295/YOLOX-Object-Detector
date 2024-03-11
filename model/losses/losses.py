import torch
import torch.nn as nn

class YOLOXLoss(nn.Module):
    def __init__(self, reg_weight=5.0, reduction='none', loss_type='giou', num_classes=6, num_fg=3 ,use_l1_loss=True):
        super(YOLOXLoss, self).__init__()
        self.reg_weight = reg_weight  # Regularization weight for IOU loss
        self.reduction = reduction  # Reduction method for loss computation
        self.loss_type = loss_type  # Type of loss: 'iou' or 'giou'
        self.use_l1_loss = use_l1_loss  # Whether to use L1 loss for bbox regression
        self.num_classes = num_classes
        self.num_fg = num_fg

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        # Separate predictions into bbox, confidence, and class predictions
        pred_bbox = pred[:, :, :4]
        pred_conf = pred[:, :, 4:5]
        pred_cls = pred[:, :, 5:]

        # Separate targets into bbox, objectness, and class labels
        target_bbox = target[:, :, :4]
        target_conf = target[:, :, 4:5]
        target_cls = target[:, :, 5:]

        self.num_fg = max(self.num_fg, 1)
        
        # Compute IOU loss
        iou_loss = self.compute_iou_loss(pred_bbox, target_bbox).sum() / self.num_fg

        # Compute confidence loss (binary cross-entropy)
        conf_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)(pred_conf, target_conf).sum() / self.num_fg
        
        # Compute classification loss (binary cross-entropy)
        cls_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)(pred_cls, target_cls).sum() / self.num_fg
        
        # Compute L1 loss for bbox regression if specified
        l1_loss = 0
        if self.use_l1_loss:
            l1_loss = nn.L1Loss(reduction=self.reduction)(pred_bbox, target_bbox).sum() / self.num_fg

        # Combine losses based on reduction type
        loss = self.reg_weight * iou_loss + conf_loss + cls_loss + l1_loss

        return (
            loss,
            self.reg_weight * iou_loss,
            conf_loss,
            cls_loss,
            l1_loss,
        )

    def compute_iou_loss(self, pred_bbox, target_bbox):
        # Compute IOU between predicted and target bounding boxes
        
        # Calculate the top-left and bottom-right coordinates of the intersection rectangle
        tl = torch.max(pred_bbox[:, :, :2], target_bbox[:, :, :2])
        br = torch.min(pred_bbox[:, :, 2:], target_bbox[:, :, 2:])
        
        # Calculate the areas of the predicted and ground truth boxes
        area_p = torch.prod(pred_bbox[:, :, 2:] - pred_bbox[:, :, :2], 2)
        area_g = torch.prod(target_bbox[:, :, 2:] - target_bbox[:, :, :2], 2)
        
        # Calculate the area of intersection, union, and IoU
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        # Calculate the loss based on the loss type
        if self.loss_type == 'giou':
            # Compute GIOU if specified
            c_tl = torch.min(pred_bbox[:, :, :2], target_bbox[:, :, :2])
            c_br = torch.max(pred_bbox[:, :, 2:], target_bbox[:, :, 2:])
            area_c = torch.prod(c_br - c_tl, 2)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            # Otherwise, compute IOU loss
            loss =  1 - iou ** 2
            
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss
        

# Example target tensor for a detection task with bounding box coordinates, objectness scores, and class labels
# Batch size = 2, number of bounding boxes = 3, each box has 4 coordinates (x1, y1, x2, y2),
# objectness score, and class label
target = torch.tensor([[[10, 20, 50, 80, 0.9, 3],   
                        [30, 40, 70, 90, 0.8, 2],   
                        [60, 70, 100, 120, 0.95, 1]],
                       [[15, 25, 55, 85, 0.85, 2],   
                        [35, 45, 75, 95, 0.7, 1],   
                        [65, 75, 105, 125, 0.92, 3]]]) 

# Example predicted tensor with bounding box coordinates, objectness scores, and class labels
# These could be the predicted outputs from your model
pred = torch.tensor([[[12, 22, 48, 78, 0.92, 3],   
                      [32, 42, 72, 92, 0.85, 2],   
                      [58, 68, 98, 118, 0.88, 1]], 
                     [[17, 27, 53, 83, 0.88, 2],  
                      [38, 48, 78, 98, 0.78, 1],   
                      [62, 72, 102, 122, 0.95, 3]]]) 

# Instantiate the YOLOXLoss with default parameters
loss_fn = YOLOXLoss()

# Compute the loss and other metrics
loss, loss_iou, loss_obj, loss_cls, loss_l1 = loss_fn(pred, target)

# Print the computed loss and metrics
print("Total Loss:", loss.item())
print("IOU Loss:", loss_iou.item())
print("Confidence Loss:", loss_obj.item())
print("Classification Loss:", loss_cls.item())
print("L1 Loss:", loss_l1.item())