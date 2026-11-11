# RetailMind AI — Incremental Machine Learning Project

RetailMind AI is an incremental retail intelligence project that explores machine learning and deep learning workflows for customer churn prediction and product-image categorisation. The project is developed in stages, with each increment adding a new model, experiment, or capability.

> **Project scope:** Days 1–5 focus on tabular customer churn prediction. Day 6 introduces a separate computer-vision module for fashion product classification. The two tracks use different datasets and independent models.

---

## Product vision

RetailMind AI aims to help businesses understand customer behaviour, identify customers at risk of leaving, and organise retail product catalogues using machine learning and AI.

The current implementation contains two modules:

1. **Customer churn prediction:** analyses telecom-style customer information and predicts whether a customer may churn.
2. **Product image classification:** uses a convolutional neural network (CNN) to classify fashion product images by their `articleType`.

---

## Project structure

```text
RetailMind_Project-main/
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── fashion-product-images-small/
│       ├── images/
│       └── styles.csv
├── models/
│   ├── churn_day1_baseline.joblib
│   ├── churn_day2_champion.joblib
│   ├── churn_day3_tuned.joblib
│   ├── churn_day4_champion.joblib
│   ├── churn_day5_ann.keras
│   ├── churn_day5_preprocessor.joblib
│   ├── fashion_mnist_prototype.keras
│   ├── vision_catalog_classifier.keras
│   ├── vision_class_names.json
│   └── vision_confusion_matrix.csv
├── reports/
├── src/
│   └── vision/
│       ├── fashion_mnist_prototype.py
│       ├── train_catalog_classifier.py
│       └── predict_catalog_image.py
├── day(1-5).ipynb
├── README.md
└── requirements.txt
```

Generated model files and downloaded datasets are local artifacts. Their presence depends on which experiments have been executed successfully.

---

# Customer churn prediction — Days 1–5

## Day 1 — Customer churn prediction baseline

### Objective

Build an end-to-end machine-learning baseline to predict whether a telecom customer is likely to leave the service.

This is a binary classification problem:

- `0` — No Churn
- `1` — Churn

The experiment uses Logistic Regression with balanced class weights. Churn recall is important because failing to identify a customer who is likely to leave can result in a missed retention opportunity.

### Dataset

Use the Telco Customer Churn dataset from Kaggle:

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the downloaded CSV in the `data/` directory. The notebook looks for:

```text
data/Telco-Customer-Churn.csv
```

It also supports:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains 7,043 customer records and 21 columns. `customerID` is removed from the predictors, and `Churn` is mapped to a binary target. `TotalCharges` is converted to numeric; 11 blank or invalid values are handled by the preprocessing pipeline.

### Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Matplotlib

### Setup and execution

Run commands from the repository root. Activate the project's Python virtual environment if it is named `base`:

```cmd
base\Scripts\activate.bat
```

Install the required packages:

```cmd
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib matplotlib jupyter
```

Launch the notebook:

```cmd
jupyter notebook
```

Open `day(1-5).ipynb` and execute cells from top to bottom.

### ML workflow

1. Load and inspect the Telco Customer Churn dataset.
2. Convert `TotalCharges` to numeric and clean the `Churn` target.
3. Remove `customerID` from the predictor columns.
4. Create a stratified 80/20 train/test split using random state 42.
5. Impute and scale numeric features.
6. Impute and one-hot encode categorical features.
7. Train a class-balanced Logistic Regression pipeline.
8. Evaluate the model and save the fitted pipeline.

### Baseline model and metrics

The baseline uses `LogisticRegression(class_weight="balanced", max_iter=1000)`.

| Metric | Day 1 result |
|---|---:|
| Accuracy | 73.81% |
| Precision (Churn) | 50.43% |
| Recall (Churn) | 78.34% |
| F1-score | 61.36% |
| ROC-AUC | 84.15% |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[747, 288],
 [ 81, 293]]
```

The model detects 293 of the 374 actual churners in the test set. Its recall is useful for a first baseline, although the 288 false-positive alerts show the precision/recall trade-off.

### Model artifact

```text
models/churn_day1_baseline.joblib
```

---

## Day 2 — Advanced churn model comparison

### Objective

Compare Decision Tree, Random Forest, and Support Vector Machine (SVM) models against the Day 1 Logistic Regression baseline. The main selection priority is cross-validation recall for the churn class.

### Experiment configuration

- Test size: 20%
- Random state: 42
- Stratified train/test split
- Three-fold stratified cross-validation on the training data
- Balanced class weights
- Shared preprocessing pipeline
- Champion selection priority: mean cross-validation churn recall

### Models

| Model | Configuration |
|---|---|
| Logistic Regression | Day 1 baseline |
| Decision Tree | Balanced class weights |
| Random Forest | Balanced class weights |
| SVM | RBF kernel and balanced class weights |

### Model comparison

| Model | Test accuracy | Test churn recall | Mean CV churn recall |
|---|---:|---:|---:|
| Logistic Regression | 73.81% | 78.34% | — |
| Decision Tree | 72.82% | 48.13% | — |
| Random Forest | 78.14% | 46.79% | — |
| SVM | 74.38% | 77.27% | 77.93% |

The SVM is selected as the Day 2 champion because it has the highest mean cross-validation recall among the Day 2 candidates. Random Forest has higher test accuracy, but lower churn recall.

### Model artifact

```text
models/churn_day2_champion.joblib
```

---

## Day 3 — Hyperparameter tuning and learning curve

### Objective

Improve the classical churn models through hyperparameter tuning and inspect how model performance changes as the amount of training data increases.

### Implementation

1. Define parameter grids for Decision Tree, Random Forest, and SVM.
2. Run `GridSearchCV` with three-fold stratified cross-validation.
3. Use churn-class recall as the search scoring metric.
4. Select the model with the highest mean cross-validation recall.
5. Evaluate the tuned model on the same held-out test set.
6. Calculate and plot training and cross-validation recall at increasing training-set sizes.

### Tuning results

| Model | Best mean CV recall |
|---|---:|
| SVM | 81.74% |
| Decision Tree | 77.93% |
| Random Forest | 72.78% |

The best SVM configuration recorded in the notebook is:

```python
C=0.5
gamma="auto"
```

### Tuned SVM test metrics

| Metric | Day 3 result |
|---|---:|
| Accuracy | 72.39% |
| Precision (Churn) | 48.78% |
| Recall (Churn) | 80.48% |
| F1-score | 60.75% |
| ROC-AUC | 83.30% |

Confusion matrix:

```text
[[719, 316],
 [ 73, 301]]
```

The tuned SVM achieves the strongest held-out churn recall among the churn models recorded in the Days 1–5 notebook: 80.48%.

### Learning curve

The notebook records the following recall values for increasing training-set sizes:

| Training examples | Training recall | Cross-validation recall |
|---:|---:|---:|
| 1,878 | 83.14% | 80.60% |
| 2,817 | 83.22% | 81.81% |
| 3,756 | 82.21% | 81.74% |

The training and cross-validation values are relatively close for the evaluated sample sizes, suggesting no large generalisation gap in this experiment.

### Generated files

```text
models/churn_day3_tuned.joblib
reports/day3_cv_tuning_results.csv
reports/day3_learning_curve.csv
reports/day3_learning_curve.png
```

---

## Day 4 — Feature engineering and feature selection

### Objective

Create additional customer-behaviour features and compare feature-selected ensemble models against the tuned Day 3 SVM.

### Feature engineering

The notebook adds four features:

- `tenure_group`
- `avg_charge_per_month`
- `is_month_to_month`
- `is_electronic_check`

The feature count increases from 19 to 23 before preprocessing.

### Model and feature-selection workflow

1. Apply the shared numeric and categorical preprocessing.
2. Use `SelectKBest` with `f_classif` to select the top 30 transformed features.
3. Train Random Forest and XGBoost candidates.
4. Compare cross-validation recall and held-out test metrics.
5. Promote the candidate with the highest mean cross-validation recall.

Feature selection is included inside the model pipeline so it is fitted within each cross-validation fold.

### Day 4 model results

| Model | Test accuracy | Precision | Churn recall | F1-score | ROC-AUC | Mean CV recall |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest + SelectKBest | 77.36% | 56.69% | 62.30% | 59.36% | 82.57% | 62.54% |
| XGBoost + SelectKBest | 79.99% | 65.75% | 51.34% | 57.66% | 84.14% | 51.71% |

Random Forest is promoted as the Day 4 champion because its mean cross-validation recall (62.54%) is higher than XGBoost's (51.71%). XGBoost records the higher test accuracy and ROC-AUC, but does not detect as many churners.

### Model artifact

```text
models/churn_day4_champion.joblib
```

### Generated reports

```text
reports/day4_vs_day3_comparison.csv
```

---

## Day 5 — Neural-network experiments

### Objective

Introduce neural networks into the churn workflow, first using a small MNIST handwritten-digit warm-up and then training a dense Artificial Neural Network (ANN) on the customer churn data.

### Technologies

- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn preprocessing and evaluation
- Matplotlib

### Experiment A — MNIST warm-up

A small neural network is trained on the built-in MNIST handwritten-digit dataset for three epochs.

| Dataset | Epochs | Test accuracy |
|---|---:|---:|
| MNIST handwritten digits | 3 | 96.84% |

This is a separate digit-classification warm-up. It is not a customer churn metric and is not the Fashion-MNIST experiment used in Day 6.

### Experiment B — Customer churn ANN

The customer data is transformed using the shared preprocessing pipeline. The 19 original input columns become 45 transformed features after numeric processing and categorical one-hot encoding.

The ANN architecture uses:

- Dense hidden layer with 64 units
- Dense hidden layer with 32 units
- Dropout regularisation
- Sigmoid output for binary classification
- Class weights to account for class imbalance
- Early stopping based on validation loss

Training was configured for up to 100 epochs. Early stopping ended the run after 18 epochs, with the best validation-loss epoch recorded as epoch 10.

### ANN test metrics

| Metric | Day 5 result |
|---|---:|
| Accuracy | 75.80% |
| Precision (Churn) | 53.14% |
| Recall (Churn) | 74.60% |
| F1-score | 62.07% |
| ROC-AUC | 83.95% |

Confusion matrix:

```text
[[789, 246],
 [ 95, 279]]
```

The ANN has higher test recall, F1-score, and ROC-AUC than the Day 4 Random Forest champion in this run, but lower accuracy and precision. It does not exceed the Day 3 tuned SVM's test churn recall.

### Model artifacts

```text
models/churn_day5_ann.keras
models/churn_day5_preprocessor.joblib
```

### Generated reports

```text
reports/day5_ann_loss_curve.png
reports/day5_ann_vs_phase1_champion.csv
reports/day5_ann_vs_phase1_champion.png
```

---

## Consolidated customer churn results — Days 1–5

The following table contains held-out test metrics recorded in the notebook. Mean cross-validation recall is shown separately where available.

| Experiment | Accuracy | Precision | Churn recall | F1-score | ROC-AUC | Mean CV recall |
|---|---:|---:|---:|---:|---:|---:|
| Day 1 Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% | 84.15% | — |
| Day 2 SVM | 74.38% | 51.15% | 77.27% | 61.55% | 82.11% | 77.93% |
| Day 3 tuned SVM | 72.39% | 48.78% | **80.48%** | 60.75% | 83.30% | 81.74% |
| Day 4 Random Forest + SelectKBest | 77.36% | 56.69% | 62.30% | 59.36% | 82.57% | 62.54% |
| Day 4 XGBoost + SelectKBest | **79.99%** | **65.75%** | 51.34% | 57.66% | **84.14%** | 51.71% |
| Day 5 Keras ANN | 75.80% | 53.14% | 74.60% | **62.07%** | 83.95% | — |

### Key observations

1. **Highest held-out churn recall:** Day 3 tuned SVM at 80.48%.
2. **Highest held-out accuracy:** Day 4 XGBoost at 79.99%.
3. **Day 4 champion:** Random Forest was promoted because its training cross-validation recall was higher than XGBoost's.
4. **Day 5 ANN:** achieved 74.60% churn recall and 83.95% ROC-AUC.
5. Accuracy alone does not determine the best churn model. The appropriate model depends on the business cost of missed churners, false-positive alerts, and retention outreach.

---


# RetailMind AI — Day 6: CNN Product Image Classification

## Goal

Add the first computer-vision module to RetailMind AI. The module classifies product images into catalog categories (`articleType`) so the catalog can be auto-tagged. This can later support visual search and other image-based retail features.

This increment uses TensorFlow/Keras and a CNN. It has two stages:

1. **Fashion-MNIST prototype:** quickly validate a CNN architecture on the built-in Keras dataset.
2. **Fashion Product Images Small:** train a catalog classifier on real product photographs.

## Files to create

Add these files to the existing repository:

```text
RetailMind_Project-main/
├── data/
│   └── fashion-product-images-small/   # download/extract locally; do not commit dataset
│       ├── images/
│       │   ├── 15970.jpg
│       │   └── ...
│       └── styles.csv
├── models/
│   ├── fashion_mnist_prototype.keras   # generated by prototype script
│   ├── vision_catalog_classifier.keras # generated by real-data training
│   ├── vision_class_names.json          # generated class-label mapping
│   └── vision_confusion_matrix.csv      # generated confusion matrix
└── src/
    └── vision/
        ├── __init__.py
        ├── fashion_mnist_prototype.py
        ├── train_catalog_classifier.py
        └── predict_catalog_image.py
```

Keep your existing churn/ANN files. This is a new module under `src/vision/`, not a replacement for previous project work.

## Dataset setup

Download **Fashion Product Images Small** from Kaggle:

`paramaggarwal/fashion-product-images-small`

Extract it so the folder contains:

- `data/fashion-product-images-small/images/` — product image files named `<id>.jpg`
- `data/fashion-product-images-small/styles.csv` — metadata containing `id` and `articleType` columns

The training script uses `articleType` as the classification label. It skips missing image files and categories with fewer than three images so each retained category can be represented in train, validation, and test sets.

Do not upload the full image dataset to GitHub. It is large and is not source code.

## Environment setup (Windows CMD)

The project uses a standard Python virtual environment named `base` (this is **not** a Conda environment). Run these commands from the repository root:

```cmd
base\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install tensorflow Pillow
```

`Pillow` is required by the image-loading utilities used by the scripts. TensorFlow installs its required NumPy dependency. If the packages are already installed in this environment, you do not need to reinstall them. Verify the imports:

```cmd
python -c "import tensorflow as tf; from PIL import Image; print('TensorFlow:', tf.__version__); print('Pillow: OK')"
```

Run all commands below from the repository root, with the `base` environment activated. If your virtual environment folder has a different name, replace `base` in the activation command with that folder name.

## Run the Fashion-MNIST prototype first

From the repository root:

```cmd
python -m src.vision.fashion_mnist_prototype
```

This script:
- loads Fashion-MNIST through `tf.keras.datasets`;
- builds a CNN using convolution, pooling, dropout, and dense layers;
- trains with a validation split and early stopping;
- evaluates test accuracy;
- prints a 10-class confusion matrix;
- saves `models/fashion_mnist_prototype.keras`.

The first run may need internet access to download Fashion-MNIST if it is not already cached.

## Train the real catalog classifier

After placing the Kaggle dataset in the expected location:

```cmd
python -m src.vision.train_catalog_classifier
```

The script:
- reads `styles.csv` with Python's standard CSV module;
- maps each `articleType` category to an integer label;
- creates per-category train/validation/test splits;
- loads and resizes images with `tf.data`;
- applies random flip, rotation, and zoom augmentation during training;
- trains a CNN;
- uses early stopping and saves the best validation checkpoint;
- evaluates the held-out test set;
- writes a confusion matrix and class-name mapping under `models/`.

The CNN uses `128 × 128` RGB images. The number of output classes is derived from the categories found in the dataset after filtering.

## Predict one image

After training:

```cmd
python -m src.vision.predict_catalog_image data/fashion-product-images-small/images/15970.jpg
```

Replace `15970.jpg` with an image file that exists in your dataset.

The script prints the predicted category and its confidence.

## CNN architecture

The catalog model follows the requested pattern:

1. Input and pixel rescaling
2. Image augmentation
3. Convolution + max pooling
4. Dropout
5. Additional convolution + pooling blocks
6. Global average pooling
7. Dense layer + dropout
8. Softmax output for product categories

## Day 6 implementation and results

### What was implemented

Day 6 was completed in two stages, using two separate datasets and two separately trained CNN models:

1. **Fashion-MNIST prototype:** used the built-in Keras Fashion-MNIST dataset to validate the image-classification workflow and CNN architecture. The dataset contains small, grayscale clothing images across 10 classes. The prototype performs image normalization, CNN feature extraction with convolution and max-pooling layers, dropout regularization, dense classification, and evaluation on the held-out test set.
2. **Real fashion catalog classifier:** used the Fashion Product Images Small dataset downloaded from Kaggle. The training script reads product IDs and `articleType` labels from `styles.csv`, finds the corresponding JPG files, filters unusable/missing examples and categories with too few images, and creates train, validation, and test splits. Images are resized to `128 × 128` RGB. Training-time augmentation includes random horizontal flipping, rotation, and zoom. The CNN is trained to predict the product's `articleType` category. Early stopping and model checkpointing are used, and a confusion matrix is exported for evaluation.

### Accuracy achieved

| Experiment | Dataset | Training duration | Accuracy achieved |
|---|---|---:|---:|
| CNN prototype | Fashion-MNIST | As run in the prototype script | Approximately **92%** |
| Product catalog CNN | Fashion Product Images Small | **12 epochs** | Approximately **68%** |

The figures above are the results reported for this Day 6 run. The real-dataset result is an initial baseline for the catalog-classification module, not a claim that the model is production-ready. Accuracy can change with the dataset version, class balance, filtering, train/test split, random seed, and training configuration. When reporting results, identify whether the printed metric is test accuracy or validation accuracy; do not compare metrics from different splits as if they were identical.

The trained models and evaluation artifacts are written to:

- `models/fashion_mnist_prototype.keras` — Fashion-MNIST prototype model.
- `models/vision_catalog_classifier.keras` — real product catalog classifier.
- `models/vision_class_names.json` — mapping between output indices and `articleType` labels.
- `models/vision_confusion_matrix.csv` — confusion matrix for the real-data test evaluation (rows are true labels; columns are predicted labels).

The Fashion-MNIST and real-product models are independent models. The Fashion-MNIST model is only a prototype; the real-data catalog model is the Day 6 deliverable intended for product-image categorization.

## Git: what to commit

Commit the source code and documentation, but keep the downloaded dataset, generated models, and generated evaluation outputs out of Git unless your team explicitly wants to version model artifacts:

```cmd
git add src/vision README_DAY6_VISION.md .gitignore
git commit -m "Document Day 6 CNN product image classification"
git push origin main
```

Do **not** commit the full Kaggle dataset. Model files can also be large; follow the repository's existing policy for generated model artifacts.

Add these entries to `.gitignore` if they are not already present:

```gitignore
# Downloaded product-image dataset
data/fashion-product-images-small/

# Generated computer-vision model artifacts
models/fashion_mnist_prototype.keras
models/vision_catalog_classifier.keras
models/vision_confusion_matrix.csv
models/vision_class_names.json
```

If your team expects model artifacts to be versioned, remove the relevant `models/` ignore entries and use the repository's agreed artifact policy.

## Notes

- This is a CNN image-classification module, not object detection. YOLO is mentioned as a future increment, not part of this Day 6 implementation.
- Fashion-MNIST is a quick architecture prototype; the real product-image dataset is the catalog-classification task.
- Test accuracy depends on the dataset version, available images, class distribution, training duration, and hardware.
- The script uses TensorFlow/Keras plus Python standard-library modules; it does not require Flask, FastAPI, scikit-learn, or a frontend.
