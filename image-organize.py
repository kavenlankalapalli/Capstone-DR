import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
csv_path = os.path.expanduser("/Users/kalyanlankalapalli/documents/gcu/milestone-3/aptos2019-blindness-detection/train.csv")  # Path to  CSV file
images_folder = os.path.expanduser("/Users/kalyanlankalapalli/documents/gcu/milestone-3/aptos2019-blindness-detection/train_images")  # Original image folder
output_base = os.path.expanduser("/Users/kalyanlankalapalli/documents/gcu/milestone-3/aptos2019-blindness-detection/images")  # Output base directory
train_ratio = 0.8  # 80% train, 20% validation

# --- Load CSV ---
df = pd.read_csv(csv_path)  #  columns : 'id_code', 'diagnosis'
df['filename'] = df['id_code'].astype(str) + ".png"  # Or .jpg depending on format

# --- Split into Train and Validation ---
train_df, val_df = train_test_split(df, test_size=1 - train_ratio, stratify=df['diagnosis'], random_state=42)

def organize_images(df_split, split_type):
    for _, row in df_split.iterrows():
        label = str(row['diagnosis'])
        src = os.path.join(images_folder, row['filename'])
        dest_dir = os.path.join(output_base, split_type, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, row['filename'])
        try:
            shutil.copyfile(src, dest)
        except FileNotFoundError:
            print(f"File not found: {src}")

# --- Organize Train and Validation Images ---
organize_images(train_df, "train")
organize_images(val_df, "validation")

print("Dataset organized successfully!")
