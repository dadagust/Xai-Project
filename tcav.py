# Launch example
#   py -3.11 tcav.py \
#       --model_path model_best.pth \
#       --data_dir data_dir \
#       --concepts_dir concepts \
#       --output_dir tcav_results

import argparse
import os
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import transforms, datasets, models
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_image_folder(path: Path, tfm: transforms.Compose) -> torch.utils.data.Dataset:
	return datasets.ImageFolder(path, transform=tfm)


def get_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivationCapture:
	"""Utility to grab activations of an intermediate layer during forward pass."""

	def __init__(self, layer: nn.Module):
		self.hook = layer.register_forward_hook(self._hook_fn)
		self.activations = None

	def _hook_fn(self, module, inp, out):
		# Flatten to (N, features)
		self.activations = out.detach().clone().view(out.size(0), -1)

	def clear(self):
		self.activations = None

	def remove(self):
		self.hook.remove()


def collect_activations(
		dataloader: torch.utils.data.DataLoader,
		model: nn.Module,
		capturer: ActivationCapture,
		device: torch.device,
) -> np.ndarray:
	feats = []
	with torch.no_grad():
		for imgs, _ in tqdm(dataloader, desc="Collect activations"):
			imgs = imgs.to(device)
			_ = model(imgs)
			feats.append(capturer.activations.cpu().numpy())
	return np.vstack(feats)


def train_cav(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
	"""Train linear classifier; return normalized weight vector (CAV)."""
	X = np.vstack([pos, neg])
	y = np.hstack(
		[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
	)  # pos=1, neg=0
	clf = make_pipeline(
		StandardScaler(with_mean=False), SGDClassifier(loss="hinge", max_iter=1000)
	)
	clf.fit(X, y)
	w: np.ndarray = clf.named_steps["sgdclassifier"].coef_.flatten()
	# normalize
	return w / (np.linalg.norm(w) + 1e-12)


def tcav_score(
		dataloader: torch.utils.data.DataLoader,
		model: nn.Module,
		capturer: ActivationCapture,
		cav: np.ndarray,
		target_idx: int,
		device: torch.device,
) -> float:
	hits = 0
	total = 0
	cav_t = torch.from_numpy(cav).float().to(device)

	for imgs, _ in tqdm(dataloader, desc="TCAV test"):
		imgs = imgs.to(device).requires_grad_(True)
		model.zero_grad()
		out = model(imgs)  # (N, 2)
		# assume CrossEntropy; pick logit for target class
		logits = out[:, target_idx]
		logits.backward(torch.ones_like(logits))

		grads = capturer.activations  # (N, F)
		# directional derivative along CAV
		directional_deriv = (grads * cav_t).sum(dim=1)
		hits += (directional_deriv > 0).sum().item()
		total += imgs.size(0)

	return hits / total


if __name__ == "__main__":
	parser = argparse.ArgumentParser("T-CAV for skin-lesion model")
	parser.add_argument("--model_path", required=True, type=str)
	parser.add_argument("--data_dir", required=True, type=str)
	parser.add_argument("--concepts_dir", required=True, type=str)
	parser.add_argument("--layer", default="layer4", type=str)
	parser.add_argument("--target_class", default="malignant", choices=["malignant", "benign"])
	parser.add_argument("--output_dir", default="tcav_results", type=str)
	parser.add_argument("--num_random", default=200, type=int,
						help="Random counterexamples per concept")
	parser.add_argument("--seed", default=42, type=int)
	args = parser.parse_args()
	set_seed(args.seed)

	device = get_device()
	os.makedirs(args.output_dir, exist_ok=True)

	# ---------------- Model ---------------- #
	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, 2)
	model.load_state_dict(torch.load(args.model_path, map_location=device))
	model = model.to(device).eval()

	layer_module = dict(model.named_modules())[args.layer]
	capturer = ActivationCapture(layer_module)

	# ---------------- Transforms ---------------- #
	mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	tfm = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])

	# ---------------- Baseline (random) loader ---------------- #
	baseline_dataset = load_image_folder(Path(args.data_dir) / "val", tfm)
	baseline_loader = torch.utils.data.DataLoader(baseline_dataset, batch_size=64, shuffle=True)
	baseline_feats = collect_activations(baseline_loader, model, capturer, device)

	# ---------------- Concept CAVs ---------------- #
	concept_dirs = [p for p in Path(args.concepts_dir).iterdir() if p.is_dir()]
	cavs: Dict[str, np.ndarray] = {}
	for concept_path in concept_dirs:
		name = concept_path.name
		ds = load_image_folder(concept_path, tfm)
		loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
		feats = collect_activations(loader, model, capturer, device)

		idx = np.random.choice(len(baseline_feats),
							   size=min(args.num_random, len(baseline_feats)),
							   replace=False)
		neg = baseline_feats[idx]

		cavs[name] = train_cav(feats, neg)
		print(f"[CAV] {name} vector trained.")

	# ---------------- TCAV scores ---------------- #
	# Загрузим все картинки из data_dir/test и выберем целевой класс
	root_test = Path(args.data_dir) / "test"
	full_test = load_image_folder(root_test, tfm)
	target_idx = full_test.class_to_idx[args.target_class]
	indices = [i for i, (_, y) in enumerate(full_test.samples) if y == target_idx]
	test_dataset = torch.utils.data.Subset(full_test, indices)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

	for concept, cav in cavs.items():
		score = tcav_score(test_loader, model, capturer, cav, target_idx, device)
		print(f"[TCAV] {concept}: {score:.3f}")

	# ---------------- Visualization ---------------- #
	plt.figure(figsize=(8, 4))
	names = list(cavs.keys())
	vals = [tcav_score(test_loader, model, capturer, cav, target_idx, device)
			for cav in cavs.values()]
	plt.bar(names, vals)
	plt.ylabel("TCAV Score")
	plt.title(f"Concept importance for class: {args.target_class}")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig(Path(args.output_dir) / f"tcav_{args.target_class}.png", dpi=300)
	capturer.remove()
