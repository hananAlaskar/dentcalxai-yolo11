# Create a Python module with meaningful functions for visualizing annotated dental images

from pathlib import Path
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from itertools import combinations


def visualize_single_class_examples(df, main_train_path, class_colors):
    """
    Visualize the first 2 images from each class that only has one class label.
    Shows bounding boxes in class-specific colors.
    """
    # Count unique classes per image
    class_counts_per_image = df.groupby('filename')['class'].nunique()
    single_class_images = class_counts_per_image[class_counts_per_image == 1]
    single_class_df = df[df['filename'].isin(single_class_images.index)]
    unique_classes = single_class_df['class'].unique()

    for class_name in unique_classes:
        class_images = single_class_df[single_class_df['class'] == class_name].head(2).reset_index()
        plt.figure(figsize=(10, 5))

        for i, row in class_images.iterrows():
            img_path = os.path.join(main_train_path, row['filename'])
            img = mpimg.imread(img_path)
            ax = plt.subplot(1, 2, i + 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Class: {class_name}")
            annotations = single_class_df[single_class_df['filename'] == row['filename']]
            for _, ann in annotations.iterrows():
                x_min, y_min, x_max, y_max = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
                color = class_colors.get(class_name, "black")
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

        plt.suptitle(f"Images for Class: {class_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def visualize_multi_class_examples(df, main_train_path, class_colors, num_images_to_show=4):
    """
    Visualize sample images with multiple classes.
    Draws bounding boxes and class labels in class-specific colors.
    """
    class_counts_per_image = df.groupby('filename')['class'].nunique()
    multi_class_images = class_counts_per_image[class_counts_per_image > 1].sort_values(ascending=False)
    multi_class_filenames = multi_class_images.head(num_images_to_show).index
    cols = 2
    rows = math.ceil(num_images_to_show / cols)

    plt.figure(figsize=(11, 4 * rows))
    for idx, filename in enumerate(multi_class_filenames):
        img_path = os.path.join(main_train_path, filename)
        img = mpimg.imread(img_path)
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')
        annotations = df[df['filename'] == filename]
        for _, row in annotations.iterrows():
            cls = row['class']
            x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            color = class_colors.get(cls, "black")
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f"Image: {filename}", fontsize=10)
        y_text = img.shape[0] + 10
        for cls in annotations['class'].unique():
            color = class_colors.get(cls, "black")
            ax.text(10, y_text, cls, color=color, fontsize=10, weight='bold')
            y_text += 20

    plt.suptitle('Images with Multiple Classes and Bounding Boxes', fontsize=16)
    plt.tight_layout()
    plt.margins(x=1, y=1)  #
    plt.show()

def are_boxes_within_margin(box1, box2, margin):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    horizontal_overlap = ((x1_min - margin <= x2_max <= x1_max + margin) and 
                          (x2_min - margin <= x1_max <= x2_max + margin))
    vertical_overlap = ((y1_min - margin <= y2_max <= y1_max + margin) and 
                        (y2_min - margin <= y1_max <= y2_max + margin))

    return horizontal_overlap and vertical_overlap


def group_bounding_boxes(df, margin=10):
    """
    Groups bounding boxes within a given spatial margin.
    Returns groups that include more than one class.
    """
    grouped_boxes = []
    for filename in df['filename'].unique():
        image_df = df[df['filename'] == filename].reset_index(drop=True)
        boxes = image_df[['xmin', 'ymin', 'xmax', 'ymax']].values
        classes = image_df['class'].values
        groups = []

        for i, box1 in enumerate(boxes):
            added = False
            for group in groups:
                if any(are_boxes_within_margin(box1, boxes[j], margin) for j in group):
                    group.append(i)
                    added = True
                    break
            if not added:
                groups.append([i])

        for group in groups:
            unique_classes = set(classes[i] for i in group)
            if len(unique_classes) > 1:
                grouped_boxes.append({
                    'filename': filename,
                    'group_indices': group,
                    'classes': unique_classes,
                    'coordinates': [boxes[i] for i in group]
                })

    return grouped_boxes


import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from itertools import combinations
from collections import defaultdict

def visualize_grouped_boxes(grouped_boxes, main_train_path, class_colors, unique_pairs_only=False):
    """
    Visualizes grouped bounding boxes that include multiple classes.
    If unique_pairs_only is True, it will show only one image per unique class pair (unordered).
    Also prints the count of how many times each pair appears in the dataset.
    """
    pair_counts = defaultdict(int)
    pair_to_box = {}

    if unique_pairs_only:
        for box_info in grouped_boxes:
            cls_set = sorted(set(box_info['classes']))
            if len(cls_set) >= 2:
                pairs = [frozenset(pair) for pair in combinations(cls_set, 2)]
                for pair in pairs:
                    pair_counts[pair] += 1
                    # Only store the *first* image that shows this pair
                    if pair not in pair_to_box:
                        pair_to_box[pair] = box_info

        # Print counts
        print("📊 Pair Frequencies:")
        for pair, count in pair_counts.items():
            print(f"  {sorted(pair)}: {count} image(s)")

        # Only show one image per unique pair
        grouped_boxes = list(pair_to_box.values())

    num_images = len(grouped_boxes)
    cols = 2
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(12, 4 * rows))
    for idx, box_info in enumerate(grouped_boxes):
        filename = box_info['filename']
        img_path = os.path.join(main_train_path, filename)
        img = mpimg.imread(img_path)
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')

        for coord, cls in zip(box_info['coordinates'], box_info['classes']):
            x_min, y_min, x_max, y_max = coord
            color = class_colors.get(cls, "black")
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f"Image: {filename}", fontsize=10)
        y_text = img.shape[0] + 10
        for cls in sorted(set(box_info['classes'])):
            color = class_colors.get(cls, "black")
            ax.text(10, y_text, cls, color=color, fontsize=15, weight='bold')
            y_text += 20

    plt.suptitle('Grouped Bounding Boxes with Multiple Classes', fontsize=16)
    plt.tight_layout()
    plt.show()


# def visualize_grouped_boxes(grouped_boxes, main_train_path, class_colors, unique_pairs_only=False):
#     """
#     Visualizes grouped bounding boxes that include multiple classes.
#     If unique_pairs_only is True, it will show only unique class pairs across images (no duplication).
#     """
#     if unique_pairs_only:
#         seen_pairs = set()
#         filtered_boxes = []
#         for box_info in grouped_boxes:
#             cls_set = sorted(set(box_info['classes']))
#             if len(cls_set) >= 2:
#                 # Generate all unique unordered class pairs
#                 pairs = list(combinations(cls_set, 2))
#                 # Check if any of the pairs are unseen
#                 new_pairs = [pair for pair in pairs if frozenset(pair) not in seen_pairs]
#                 if new_pairs:
#                     filtered_boxes.append(box_info)
#                     for pair in new_pairs:
#                         seen_pairs.add(frozenset(pair))
#         grouped_boxes = filtered_boxes

#     num_images = len(grouped_boxes)
#     cols = 2
#     rows = math.ceil(num_images / cols)

#     plt.figure(figsize=(12, 4 * rows))
#     for idx, box_info in enumerate(grouped_boxes):
#         filename = box_info['filename']
#         img_path = os.path.join(main_train_path, filename)
#         img = mpimg.imread(img_path)
#         ax = plt.subplot(rows, cols, idx + 1)
#         ax.imshow(img)
#         ax.axis('off')
#         for coord, cls in zip(box_info['coordinates'], box_info['classes']):
#             x_min, y_min, x_max, y_max = coord
#             color = class_colors.get(cls, "black")
#             rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                                      linewidth=2, edgecolor=color, facecolor='none')
#             ax.add_patch(rect)

#         ax.set_title(f"Image: {filename}", fontsize=10)
#         y_text = img.shape[0] + 10
#         for cls in sorted(set(box_info['classes'])):
#             color = class_colors.get(cls, "black")
#             ax.text(10, y_text, cls, color=color, fontsize=15, weight='bold')
#             y_text += 20

#     plt.suptitle('Grouped Bounding Boxes with Multiple Classes', fontsize=16)
#     plt.tight_layout()
#     plt.show()
