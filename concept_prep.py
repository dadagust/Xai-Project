# Launch example:
#   py -3.11 concept_prep_all.py \
#       --metadata archive/HAM10000_metadata.csv \
#       --images_dir archive/HAM10000_images \
#       --out_dir concepts

import argparse
import shutil
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--metadata", required=True, type=str)
	p.add_argument("--images_dir", required=True, type=str)
	p.add_argument("--out_dir", default="concepts", type=str)
	args = p.parse_args()

	df = pd.read_csv(args.metadata)
	img_dir = Path(args.images_dir)
	out_root = Path(args.out_dir)
	out_root.mkdir(exist_ok=True)

	# Столбцы, по которым делаем концепты
	skip_cols = {"image_id", "lesion_id"}
	columns = [c for c in df.columns if c not in skip_cols]

	total = 0
	for col in columns:
		unique_vals = df[col].dropna().unique()
		for val in unique_vals:
			concept_name = f"{col}_{val}"
			target_dir = out_root / concept_name / "0"
			target_dir.mkdir(parents=True, exist_ok=True)

			subset = df[df[col] == val]
			copied = 0
			for _, row in subset.iterrows():
				src = img_dir / f"{row.image_id}.jpg"
				dst = target_dir / src.name
				if src.is_file() and not dst.exists():
					shutil.copyfile(src, dst)
					copied += 1
			print(f"{concept_name:25s}: {copied:5d} files")
			total += copied

	print(f"\nTotal images copied: {total}")
