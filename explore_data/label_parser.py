# File: label_parser.py
"""
Label Parsing Utilities for Multi-Label Dental Dataset (YOLO/Roboflow Format)

Supports:
- Loading annotation data from flat YOLO-style CSVs
- Extracting label stats
- Filtering by label count
- Grouping overlapping boxes (e.g., per tooth)
"""

import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from shapely.geometry import box as shapely_box

def load_annotations_csv(csv_path: str) -> pd.DataFrame:
    """
    Load YOLO-style annotation CSV with columns:
    ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    Groups them per image into 'Labels' and 'BBoxes'.
    """
    df_raw = pd.read_csv(csv_path)
    grouped = df_raw.groupby("filename")
    records = []
    for img_name, group in grouped:
        labels = group["class"].tolist()
        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
        records.append({"Image": img_name, "Labels": labels, "BBoxes": boxes})
    return pd.DataFrame(records)

def get_class_frequencies(df: pd.DataFrame) -> pd.Series:
    all_labels = [label for labels in df['Labels'] for label in labels]
    return pd.Series(all_labels).value_counts()

def filter_images_by_num_classes(df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    return df[df['Labels'].apply(lambda x: len(set(x)) == num_classes)].reset_index(drop=True)

def build_image_label_index(df: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple[int, int, int, int]]]]:
    img_index = {}
    for _, row in df.iterrows():
        image = row['Image']
        labels = row['Labels']
        boxes  = row['BBoxes']
        label_map = defaultdict(list)
        for label, box in zip(labels, boxes):
            label_map[label].append(tuple(box))
        img_index[image] = dict(label_map)
    return img_index

def group_boxes_per_teeth(bboxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.3) -> List[List[Tuple[int, int, int, int]]]:
    groups = []
    used = [False] * len(bboxes)

    for i, b1 in enumerate(bboxes):
        if used[i]:
            continue
        group = [b1]
        used[i] = True
        b1_shape = shapely_box(*b1)

        for j, b2 in enumerate(bboxes):
            if i == j or used[j]:
                continue
            b2_shape = shapely_box(*b2)
            iou = b1_shape.intersection(b2_shape).area / b1_shape.union(b2_shape).area
            if iou >= iou_thresh:
                group.append(b2)
                used[j] = True

        groups.append(group)
    return groups

def get_teeth_with_multiple_labels(img_index: Dict[str, Dict[str, List[Tuple[int, int, int, int]]]], iou_thresh=0.3) -> Dict[str, List[Tuple[List[str], List[Tuple[int, int, int, int]]]]]:
    results = {}
    for image, label_dict in img_index.items():
        all_boxes = []
        label_for_box = []
        for label, boxes in label_dict.items():
            for b in boxes:
                all_boxes.append(b)
                label_for_box.append(label)

        groups = group_boxes_per_teeth(all_boxes, iou_thresh)
        multi_label_groups = []
        for group in groups:
            labels_in_group = list({label_for_box[all_boxes.index(b)] for b in group})
            if len(labels_in_group) > 1:
                multi_label_groups.append((labels_in_group, group))

        if multi_label_groups:
            results[image] = multi_label_groups

    return results

def expand_bbox(bbox: Tuple[int, int, int, int], margin: float = 0.1, image_size: Tuple[int, int] = (1024, 1024)) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * margin), int(h * margin)
    new_x1 = max(0, x1 - dx)
    new_y1 = max(0, y1 - dy)
    new_x2 = min(image_size[0], x2 + dx)
    new_y2 = min(image_size[1], y2 + dy)
    return (new_x1, new_y1, new_x2, new_y2)
