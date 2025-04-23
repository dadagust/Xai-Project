# Launch example:
#   py -3.11 improvements.py \
#       --metadata archive/HAM10000_metadata.csv \
#       --images_dir archive/HAM10000_images \
#       --data_dir data_dir \
#       --concepts_dir concepts \
#       --model model_best.pth

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def set_seed(seed=42):
	random.seed(seed);
	np.random.seed(seed);
	torch.manual_seed(seed)


class ActivationCapture:
	def __init__(self, layer):
		self.hook = layer.register_forward_hook(self.h);
		self.act = None

	def h(self, m, i, o): self.act = o.detach().view(o.size(0), -1)

	def remove(self): self.hook.remove()


def load_imgfolder(path, tfm):
	return datasets.ImageFolder(path, transform=tfm)


def collect_feats(loader, model, capt):
	feats = []
	with torch.no_grad():
		for x, _ in loader:
			_ = model(x.to(device))
			feats.append(capt.act.cpu().numpy())
	return np.vstack(feats)


def train_cav(pos, neg):
	X = np.vstack([pos, neg])
	y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
	from sklearn.linear_model import SGDClassifier
	clf = SGDClassifier(loss="hinge", max_iter=1000)
	pipe = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-9)
	clf.fit(pipe(X), y)
	w = clf.coef_.flatten()
	return w / np.linalg.norm(w)


def tcav_score(loader, model, capt, cav, cls_idx):
	cav_t = torch.from_numpy(cav).float().to(device)
	hits = total = 0
	with torch.enable_grad():
		for x, _ in loader:
			x = x.to(device)
			x.requires_grad_(True)
			model.zero_grad()
			logits = model(x)[:, cls_idx].sum()
			logits.backward()
			dd = (capt.act * cav_t).sum(1)
			hits += (dd > 0).sum().item()
			total += x.size(0)
	return hits / total


def tcav_bootstrap(cav, loader, model, capt, cls_idx, iters=500):
	score = tcav_score(loader, model, capt, cav, cls_idx)
	cav_np = cav.copy()

	grads = []
	with torch.enable_grad():
		for x, _ in loader:
			x = x.to(device);
			x.requires_grad_(True)
			model.zero_grad()
			model(x)[:, cls_idx].sum().backward()
			grads.append(capt.act.cpu())
	grads = torch.cat(grads)  # (N, F)

	samples = []
	for _ in range(iters):
		sign = np.random.choice([-1, 1], len(cav_np))
		cav_rand = torch.from_numpy(cav_np * sign).float()
		dd = (grads @ cav_rand).numpy()
		samples.append((dd > 0).mean())

	p_val = (np.sum(np.array(samples) >= score) + 1) / (iters + 1)
	return score, p_val

# Actually not needed, but anyway:
def md_report(stats, fig_paths, out_md):
	with open(out_md, 'w', encoding='utf8') as f:
		f.write("# TCAV Analysis Report\n")
		for cls, data in stats.items():
			f.write(f"\n## {cls}\n\n| Concept | Score | p-value |\n|---|---|---|\n")
			for c, (s, p) in sorted(data.items(), key=lambda x: -x[1][0]):
				f.write(f"| {c} | {s:.3f} | {p:.4f} |\n")
			f.write(f"\n![]({fig_paths[cls]})\n")


if __name__ == "__main__":
	""" 
	    Pipeline :
		 1. Cav collection for all concepts
		 2. TCAV-score + bootstrap-p-value for classes malignant/benign
		 3. Markdown generation
		 4. Balanced DataLoader and adversarial-debias example
		 5. Creation of csv without dx_type
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--metadata", required=True)
	parser.add_argument("--images_dir", required=True)
	parser.add_argument("--data_dir", required=True)
	parser.add_argument("--concepts_dir", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--layer", default="layer4")
	parser.add_argument("--output_dir", default="tcav_results")
	args = parser.parse_args()
	set_seed(42)
	os.makedirs(args.output_dir, exist_ok=True)

	# csv without dx_type
	df = pd.read_csv(args.metadata).drop(columns=['dx_type'], errors='ignore')
	df.to_csv("metadata_nodxtype.csv", index=False)

	# model
	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, 2)
	model.load_state_dict(torch.load(args.model, map_location=device))
	model.to(device).eval()
	capt = ActivationCapture(dict(model.named_modules())[args.layer])

	mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
							  transforms.ToTensor(), transforms.Normalize(mean, std)])

	# 2. baseline features (val split)
	base_ds = load_imgfolder(Path(args.data_dir) / "val", tfm)
	base_ld = DataLoader(base_ds, batch_size=64, shuffle=True)
	base_feats = collect_feats(base_ld, model, capt)

	cavs = {}
	for concept_dir in Path(args.concepts_dir).iterdir():
		if not concept_dir.is_dir(): continue
		pos_ds = load_imgfolder(concept_dir, tfm)
		pos_ld = DataLoader(pos_ds, batch_size=64, shuffle=False)
		pos_feats = collect_feats(pos_ld, model, capt)
		neg_idx = np.random.choice(len(base_feats), min(len(base_feats), 200), replace=False)
		cavs[concept_dir.name] = train_cav(pos_feats, base_feats[neg_idx])
		print(f"[CAV] {concept_dir.name} ready ({len(pos_ds)})")

	tcav_stats = {"malignant": {}, "benign": {}}
	class_map = {'benign': 0, 'malignant': 1}

	for cls in ['malignant', 'benign']:
		full_test = load_imgfolder(Path(args.data_dir) / "test", tfm)
		idx_cls = class_map[cls]
		sel_idx = [i for i, (_, y) in enumerate(full_test.samples) if y == idx_cls]
		test_ld = DataLoader(Subset(full_test, sel_idx), batch_size=32, shuffle=False)

		for c, v in cavs.items():
			s, p = tcav_bootstrap(v, test_ld, model, capt, idx_cls, iters=500)
			tcav_stats[cls][c] = (s, p)
			print(f"[{cls}] {c:25s} score={s:.3f} p={p:.4f}")

		# save bar-chat
		plt.figure(figsize=(10, 4))
		names, vals = zip(*[(n, s) for n, (s, _) in tcav_stats[cls].items()])
		plt.bar(names, vals);
		plt.xticks(rotation=90);
		plt.ylabel("TCAV")
		plt.title(f"{cls} concept importance");
		plt.tight_layout()
		fig_path = Path(args.output_dir) / f"tcav_{cls}.png"
		plt.savefig(fig_path, dpi=300);
		plt.close()

	# 6. Markdown
	fig_paths = {cls: str(Path(args.output_dir) / f"tcav_{cls}.png") for cls in tcav_stats}
	md_report(tcav_stats, fig_paths, "TCAV_Report.md")
