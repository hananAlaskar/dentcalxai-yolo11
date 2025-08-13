# Create a modular Python script for extracting, counting, and plotting class distributions

import os
import pandas as pd
import matplotlib.pyplot as plt


def extract_csv_to_dataframe(directory):
    """
    Extracts all .csv files from subfolders and stores them in a nested dictionary.
    Format: dataframes[folder][filename] = DataFrame
    """
    dataframes = {}
    for root, dirs, files in os.walk(directory):
        if root != directory:
            folder_name = os.path.basename(root)
            folder_data = {}
            csv_files = [f for f in files if f.lower().endswith('.csv') and not f.startswith('.')]
            for csv_file in csv_files:
                path = os.path.join(root, csv_file)
                try:
                    df = pd.read_csv(path)
                    df_key = os.path.splitext(csv_file)[0]
                    folder_data[df_key] = df
                except Exception as e:
                    print(f"⚠️ Error reading {csv_file}: {e}")
            if folder_data:
                dataframes[folder_name] = folder_data

    # Sort folders: 'train' comes first, then alphabetically
    sorted_folders = sorted(dataframes.keys(), key=lambda x: (x != "train", x))
    return {folder: dataframes[folder] for folder in sorted_folders}


def count_classes_and_images(dataframes):
    """
    Counts total and per-image occurrences of each class in each folder.
    Returns a dictionary of structure:
    class_counts[folder][class_name] = { 'total_count': int, 'images_count': set(...) }
    """
    class_counts = {}
    for folder_name, folder_data in dataframes.items():
        folder_class_counts = {}
        for file_name, df in folder_data.items():
            for _, row in df.iterrows():
                class_name = row['class']
                image_name = row['filename']
                if class_name not in folder_class_counts:
                    folder_class_counts[class_name] = {
                        'total_count': 0,
                        'images_count': set()
                    }
                folder_class_counts[class_name]['total_count'] += 1
                folder_class_counts[class_name]['images_count'].add(image_name)
        class_counts[folder_name] = folder_class_counts
    return class_counts




def plot_class_distribution(class_counts, fixed_class_order):
    """
    Plots bar charts of class and image frequency for each folder (e.g., train/val/test).
    Also prints total and image-level counts for each class.
    """
    for folder_name, folder_class_counts in class_counts.items():
        total_counts = [folder_class_counts.get(cls, {}).get('total_count', 0) for cls in fixed_class_order]
        image_counts = [len(folder_class_counts.get(cls, {}).get('images_count', set())) for cls in fixed_class_order]

        # 🔢 Print raw count info
        print(f"\n📂 Folder: {folder_name}")
        print("Total box count per class:", total_counts)
        print("Image count per class     :", image_counts)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Total class count ---
        bars_total = axes[0].bar(fixed_class_order, total_counts, color='lightblue')
        axes[0].set_title(f"Box Count in {folder_name}")
        axes[0].set_ylabel("Total Instances")
        for bar in bars_total:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, height + 1, f"{int(height)}", 
                         ha='center', va='bottom', fontsize=10)

        # --- Unique image count ---
        bars_images = axes[1].bar(fixed_class_order, image_counts, color='lightcoral')
        axes[1].set_title(f"Image Count in {folder_name}")
        axes[1].set_ylabel("Unique Images")
        for bar in bars_images:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, height + 1, f"{int(height)}", 
                         ha='center', va='bottom', fontsize=10)

        for ax in axes:
            ax.set_xlabel("Class")
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


# def plot_class_distribution(class_counts, fixed_class_order):
#     """
#     Plots bar charts of class and image frequency for each folder (e.g., train/val/test).
#     """
#     for folder_name, folder_class_counts in class_counts.items():
#         total_counts = [folder_class_counts.get(cls, {}).get('total_count', 0) for cls in fixed_class_order]
#         image_counts = [len(folder_class_counts.get(cls, {}).get('images_count', set())) for cls in fixed_class_order]

#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#         # Total class count bar
#         axes[0].bar(fixed_class_order, total_counts, color='lightblue')
#         axes[0].set_title(f"Box Count in {folder_name}")
#         axes[0].set_ylabel("Total Instances")
#         for i, val in enumerate(total_counts):
#             axes[0].text(i, val + 0.1, str(val), ha='center', fontsize=10)

#         # Unique image count bar
#         axes[1].bar(fixed_class_order, image_counts, color='lightcoral')
#         axes[1].set_title(f"Image Count in {folder_name}")
#         axes[1].set_ylabel("Unique Images")
#         for i, val in enumerate(image_counts):
#             axes[1].text(i, val + 0.1, str(val), ha='center', fontsize=10)

#         for ax in axes:
#             ax.set_xlabel("Class")
#             ax.tick_params(axis='x', rotation=45)

#         plt.tight_layout()
#         plt.show()

