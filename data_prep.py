import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

csv_path = 'archive/HAM10000_metadata.csv'
images_dir = 'archive/HAM10000_images'
output_dir = 'data_dir'
val_size = 0.1
test_size = 0.1
random_seed = 42

malignant_labels = ['mel', 'bcc', 'akiec']
benign_labels = ['nv', 'bkl', 'df', 'vasc']

label_map = {
    **{lbl: 'malignant' for lbl in malignant_labels},
    **{lbl: 'benign' for lbl in benign_labels},
}

df = pd.read_csv(csv_path)
df['path'] = df['image_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
df['label'] = df['dx'].map(label_map)
df = df[df['label'].notnull()]

train_df, testval_df = train_test_split(df, test_size=val_size + test_size,
                                        stratify=df['label'], random_state=random_seed)
val_df, test_df = train_test_split(testval_df, test_size=test_size / (test_size + val_size),
                                   stratify=testval_df['label'], random_state=random_seed)

splits = [('train', train_df), ('val', val_df), ('test', test_df)]

for split_name, split_df in splits:
    for label in ['benign', 'malignant']:
        Path(output_dir, split_name, label).mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        src = row['path']
        dst = Path(output_dir, split_name, row['label'], os.path.basename(src))
        shutil.copyfile(src, dst)

print(f"[✓] Data saved into: {output_dir}")
