---
weight: 1
bookFlatSection: true
title: "Convolutional Neural Network (CNN) in the domain of dermatology"
---

<style> .markdown a{text-decoration: underline !important;} </style>
<style> .markdown h2{font-weight: bold;} </style>

# *Convolutional Neural Network (CNN) in the domain of dermatology*

**Authors: Nikita Kurkulskiu**

*To see the implementation, visit [github project](https://github.com/dadagust/Xai-Project).*

<!-- Example of image loading -->
<!-- ![Diffusion Lens Diagram](/Diffusion%20Lens:%20Interpreting%20Text%20Encoders%20in%20Text-to-Image%20pipelines%20tuned%20using%20DreamBooth/dreambooth.png) -->

# *Convolutional Neural Network (CNN) in the domain of dermatology*

**Author:** Nikita Kurkulskiu  
*Full implementation on [GitHub](https://github.com/dadagust/Xai-Project).*

---

## **Introduction**

Skin-cancer screening with deep learning can reduce diagnostic workload, but
black-box models hinder clinical adoption.  
We fine-tune a ResNet-18 to separate **melanoma** from **benign lesions** on
the open **HAM10000** dermatoscopic dataset, then make its decisions
*explainable and auditable* with **T-CAV (Testing with Concept Activation
Vectors)**.  T-CAV quantifies how much *human-defined concepts*
(age, sex, body location, diagnosis labels) drive predictions, exposing hidden
bias and validating clinically relevant factors.

---

## **Background**

Classical explainability (Grad-CAM, LRP) highlights *where* a network looks, but
does not answer *what* semantic attributes it uses.  
T-CAV fills this gap:

1. **Concept definition** – A user provides a small image set exemplifying a
   concept (e.g., *“male patients”*).
2. **CAV learning** – A linear separator in latent space defines a
   *Concept Activation Vector*.
3. **TCAV score** – The directional derivative of the class logit along the CAV
   is computed; the proportion of **positive** derivatives estimates concept
   influence.
4. **Bootstrap p-value** – Random sign-flipped CAVs test statistical
   significance.

---

### **Section 1 – Dataset**

* **HAM10000** (Kaggle): 10 015 dermatoscopic JPEGs across 7 diagnostic
  categories.  
* We convert metadata into >100 concepts (age buckets, sex, localisation,
  diagnosis, etc.) using `concept_prep.py`.

### **Section 2 – Ethical Motivation**

Demographic or anatomical biases in datasets can translate into unfair or
unsafe predictions.  Quantifying concept influence lets us:

* Detect **gender/age bias**.
* Validate model focus on clinical features instead of artefacts (e.g.,
  histology stamp).

---

## **Methodology**

### **3.1 Code Organisation**

```text
Xai-Project/
├── archive/               # Raw HAM10000 images + metadata CSV
├── data_dir/              # train/val/test split (benign vs malignant)
├── concepts/              # >100 sub-folders with concept images
├── tcav_results/          # Auto-generated bar-charts & report
├── train.py               # CNN training script
├── data_prep.py           # Builds data_dir/
├── concept_prep.py        # Builds concepts/
├── tcav.py                # Basic TCAV (single class)
├── statistical_check.py   # Full pipeline: CAV, bootstrap, markdown report
└── TCAV_Report.md         # Auto-generated


```

---

### **3.1  Model & Code Snippets**


Training (excerpt from train.py)

```
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)          # 2 classes
...
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc     = evaluate(...)

```
T-CAV scoring (excerpt)
```
def tcav_score(loader, model, capt, cav, cls_idx):
    cav_t = torch.from_numpy(cav).float().to(device)
    hits = total = 0
    with torch.enable_grad():
        for x,_ in loader:
            x = x.to(device).requires_grad_(True)
            model.zero_grad()
            model(x)[:, cls_idx].sum().backward()
            dd = (capt.act * cav_t).sum(1)             # directional derivative
            hits += (dd > 0).sum().item(); total += x.size(0)
    return hits / total
```

---

### **3.3 Pipeline**

```
flowchart LR
    A[data_prep.py] --> B(train.py)
    B -->|model_best.pth| C(statistical_check.py)
    C --> D[TCAV_Report.md<br>bar charts]
    E(concept_prep.py) --> C

```

## **Insights**

### **Key TCAV Findings (bootstrap p ≤ 0.05)**

| Class       | Concept                   | Score  | p-value | Interpretation                                                                 |
|-------------|---------------------------|-------:|--------:|--------------------------------------------------------------------------------|
| **malignant** | `sex_male`               | **1.000** | **0.030** | Strong male bias – model nearly always boosts melanoma likelihood for males.   |
| malignant   | `dx_mel`                  | 0.974  | 0.056   | High influence, near significance threshold (model recognizes “melanoma”).     |
| malignant   | `localization_chest`      | 0.990  | 0.054   | Chest location strongly pushes toward malignant.                               |

_No benign concepts passed p ≤ 0.05; highest benign concept was_  
`localization_lower_extremity` (score 0.984, p = 0.090).

### **Bias & Fairness Analysis**

* **Gender:** `sex_male` TCAV = 1 → network over-predicts melanoma for men.  
* **Age:** Scores > 0.70 for age ≥ 60, but p > 0.14 – weak evidence.  
* **Location:** Chest/trunk dominate; ear/hand minimal influence, indicating imbalance.

---

## **Conclusion**

* Achieved ~90 % validation accuracy with ResNet-18 on HAM10000.  
* T-CAV uncovered **pronounced demographic and anatomical biases** beyond lesion visuals.  
* Recommended next steps:  
  1. Apply adversarial debiasing (prototype in `statistical_check.py`).  
  2. Introduce visual ABCD concepts (asymmetry, border irregularity).  
  3. Validate findings on external datasets (PH2, Derm7pt).  


