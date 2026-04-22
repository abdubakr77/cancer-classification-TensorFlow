# Breast Cancer Classification with Transfer Learning

> Classifying ultrasound breast images as **Benign** or **Malignant** using five pre-trained CNN architectures via Transfer Learning in TensorFlow / Keras.

---

## Demo Preview

![Model Predictions Grid](predictions__grid.png)

---

## Project Overview

This project applies **Transfer Learning** to a binary medical image classification task. Five well-known CNN backbones — pre-trained on ImageNet — are fine-tuned (feature extraction mode) on a breast cancer ultrasound dataset. The goal is to compare their performance and identify the best architecture for this specific domain.

### Best Result
**InceptionV3** achieved **~95% validation accuracy**, outperforming all other tested architectures.

---

## Dataset

| Property | Details |
|----------|---------|
| Name | Ultrasound Breast Cancer Images |
| Task | Binary Classification |
| Classes | `benign` / `malignant` |
| Split | Pre-split `train/` and `val/` directories |
| Format | PNG / JPG images organized in class sub-folders |

> **Data Cleaning Note:** The validation set originally contained augmented images (identified via filename regex patterns). These were filtered out to prevent data leakage — only original images are used for evaluation.

---

## Models Compared

| Model | Input Size | Params | Val Accuracy |
|-------|-----------|--------|-------------|
| **InceptionV3** | 299×299 | ~23M | **~95%** |
| VGG16 | 224×224 | ~138M | — |
| ResNet50 | 224×224 | ~25M | — |
| EfficientNetB4 | 380×380 | ~19M | — |
| MobileNetV2 | 224×224 | ~3.4M | — |

---

## Architecture

All models follow the same pattern:

```
Pre-trained Backbone (frozen)
        ↓
Global Max Pooling
        ↓
Dense(2, activation='softmax')
```

- **Backbone**: Frozen (no backbone weights updated during training)
- **Head**: Single Dense layer with Softmax output
- **Optimizer**: AdamW (lr=1e-3)
- **Loss**: Categorical Crossentropy
- **Regularization**: EarlyStopping (patience=3, monitor=val_loss, restore_best_weights=True)

---

## Project Structure

```
├── Cancer-Classification-TensorFlow.ipynb   # Main notebook
├── README.md
└── dataset/
    ├── train/
    │   ├── benign/
    │   └── malignant/
    └── val/
        ├── benign/
        └── malignant/
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdubakr77/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn
   ```

3. **Update dataset paths** in the notebook config cell:
   ```python
   TRAIN_PATH = "path/to/your/train"
   VAL_PATH   = "path/to/your/val"
   ```

4. **Run the notebook** — each `fit()` call trains and evaluates one model end-to-end.

---

## Key Implementation Details

- **Validation data leakage prevention**: Augmented images in the validation set are detected via regex filename matching and excluded before evaluation
- **Model-specific input sizes**: InceptionV3 uses 299×299, EfficientNetB4 uses 380×380, others use 224×224
- **Reproducibility**: Training shuffle uses `seed=44`
- **Evaluation**: Confusion matrix + classification report (precision, recall, F1) for each model

---

## Results

### InceptionV3 — Best Model

- **Validation Accuracy**: ~95%
- **Architecture advantage**: Inception modules capture multi-scale features simultaneously, which is beneficial for detecting subtle texture differences in ultrasound images

### Confusion Matrix (InceptionV3)

```
               Predicted
               Benign  Malignant
Actual Benign  [  TP  ] [  FN  ]
       Malignant[ FP  ] [  TN  ]
```

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-grey?logo=scikit-learn)

---

## Author

**Abdullah Bakr** — AI Engineer  
Connect on [LinkedIn](https://www.linkedin.com/in/abdubakr/) | [Kaggle](https://github.com/abdubakr77)

---

*Feel free to fork, star, or open an issue if you have suggestions!*
