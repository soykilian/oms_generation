import torch

def masks_to_boxes(masks):
    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.uint8)
    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if y.numel() == 0 or x.numel() == 0:
            bounding_boxes[index, :] = torch.zeros(4)
        else:
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
    return  bounding_boxes

def calculate_intersection_area(gt, pred):
    x2 = max(gt[0], pred[0])
    y2 = max(gt[1], pred[1])
    x1 = min(gt[2], pred[2])
    y1 = min(gt[3], pred[3])
    intersection_width = max(0, x2 - x1 + 1)
    intersection_height = max(0, y2 - y1 + 1)
    return intersection_width * intersection_height

def detection_rate(gt, pred):
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    intersection_area = calculate_intersection_area(gt, pred)
    ratio = intersection_area / gt_area
    outside_area = pred_area - intersection_area

    return 1 if ratio > 0.5 and outside_area < intersection_area else 0

