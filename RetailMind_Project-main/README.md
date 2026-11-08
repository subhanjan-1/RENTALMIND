# RetailMind AI Project installation

## Product vision
#PS C:\Retail-mind\RetailMind_Project-main>

RetailMind AI is an end-to-end retail intelligence platform that uses machine
learning and AI to help businesses understand customers, predict churn, and
improve customer retention.

## Day 1 — Customer churn prediction

### Objective

Build an end-to-end machine-learning pipeline to predict whether a telecom
customer is likely to leave the company.

### Dataset

Use the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Download the CSV and place it at `data/telco_customer_churn.csv`. The dataset is
not included in this repository.

### Technologies

- Python
- Pandas
- Scikit-learn
- Joblib

### Setup and execution

Run these commands from the repository root:

```shell
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/predict.py
```

`train.py` loads and inspects the data, cleans the target and charge columns,
creates a stratified 80/20 split, trains the preprocessing/model pipeline, and
saves it to `models/churn_pipeline.joblib`. `evaluate.py` trains the same fixed
baseline and prints its held-out metrics. `predict.py` loads the Day 2 champion
pipeline and runs a sample prediction; run `python src/model_comparison.py`
before predicting.

The model artifact and downloaded CSV are excluded by `.gitignore`. Keep or
publish them according to your dataset and artifact-storage policy.

### ML workflow

1. Data ingestion and inspection
2. Identifier removal and target/charge cleaning
3. Numeric imputation and scaling; categorical imputation and one-hot encoding
4. Stratified train/test split
5. Class-balanced logistic regression training
6. Evaluation and model persistence
7. Prediction with the persisted preprocessing/model pipeline

### Baseline model and metrics

The baseline is logistic regression with `class_weight="balanced"`,
`max_iter=1000`, a 20% stratified test split, and random state 42. Keep these
settings fixed when comparing baseline results.

| Metric | Baseline |
|---|---:|
| Accuracy | Not yet measured |
| Precision | Not yet measured |
| Recall | Not yet measured |
| ROC-AUC | Not yet measured |

After running `python src/evaluate.py` with the dataset, record the four printed
values above. They are intentionally not estimated because no dataset was
provided with this repository.

### Prediction service

`src.predict.predict()` accepts a dictionary containing the dataset's feature
columns (without `customerID` or `Churn`) and returns a prediction (`Churn` or
`No Churn`) and churn probability. It reports missing or unexpected input
features rather than silently accepting a mismatched record.

### Key concepts

1. **Why `ColumnTransformer`?** It applies numeric and categorical
   preprocessing to their respective columns within one estimator.
2. **Why `Pipeline`?** It keeps preprocessing and the classifier together,
   applies the same transformations at training and inference, and helps
   prevent data leakage by fitting transformations only on training data.
3. **Why `stratify=y`?** It preserves the churn/non-churn proportions in both
   train and test splits.
4. **`predict()` vs `predict_proba()`:** `predict()` returns a class label;
   `predict_proba()` returns estimated probabilities for each class.
5. **Why save the whole pipeline?** The persisted artifact includes both the
   fitted preprocessing and classifier, so inference uses the same learned
   transformations.
6. **Why is recall important for churn?** Higher recall finds more customers
   who may leave, reducing missed retention opportunities. It should be
   balanced against precision and the cost of retention outreach.

## Day 2 — Advanced churn model comparison

### Objective

Compare tree-based and margin-based classifiers with the Day 1 logistic
regression baseline, prioritizing recall for customers who churn.

### Experiment configuration

- Test size: 20%
- Random state: 42
- Stratified sampling: enabled
- Shared numeric/categorical preprocessing and held-out test data
- Class weights: balanced
- Champion priority: churn recall; ROC-AUC and precision break ties

Run the comparison from the repository root:

```shell
python src/model_comparison.py
python src/predict.py
```

The comparison retrains the Day 1 baseline and three Day 2 models on the same
split, reports accuracy, precision, recall, and ROC-AUC, saves each complete
pipeline, and writes the highest-recall pipeline to
`models/champion_model.joblib`.

### Models and artifacts

| Model | Configuration | Artifact |
|---|---|---|
| Logistic Regression | Day 1 baseline; `class_weight="balanced"` | `models/churn_pipeline.joblib` |
| Decision Tree | `max_depth=12`, balanced classes | `models/decision_tree.joblib` |
| Random Forest | 200 trees, `max_depth=12`, balanced classes | `models/random_forest.joblib` |
| SVM | RBF kernel, probability estimates, balanced classes | `models/svm.joblib` |
| Champion | Highest recall; ROC-AUC then precision tie-breaks | `models/champion_model.joblib` |

### Model comparison

Run `python src/model_comparison.py` with the dataset to populate these
experiment results. Values are not estimated because the dataset is not
included in this repository.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (Day 1) | Not measured | Not measured | Not measured | Not measured |
| Decision Tree | Not measured | Not measured | Not measured | Not measured |
| Random Forest | Not measured | Not measured | Not measured | Not measured |
| SVM | Not measured | Not measured | Not measured | Not measured |

The champion is selected by churn-class recall. ROC-AUC and precision are used
as tie-breakers. Accuracy alone does not determine the best model when the goal
is to identify as many at-risk customers as possible.

### Key concepts

- A decision tree learns non-linear decision rules.
- A random forest combines many decision trees to improve generalization.
- An RBF-kernel SVM can model non-linear boundaries; probability estimates are
  enabled to calculate ROC-AUC.
- `class_weight="balanced"` increases the influence of the less frequent class.
- Comparing all models on the same stratified split makes the metrics
  comparable.

## Future development

RetailMind AI will incrementally evolve into a complete retail AI platform
with additional machine-learning models, analytics, and deployment
capabilities.


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
| Product catalog CNN | Fashion Product Images Small | **12 epochs** | Approximately **56%** |

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
