# MNIST Handwritten Digit Recognition

## Objective
Build a machine learning model that looks at an image of a handwritten
digit and correctly identifies which number it is (0 through 9).

## Dataset
- **Name:** MNIST
- **Source:** OpenML (loaded via scikit-learn)
- **Total Images:** 70,000
- **Image Size:** 28x28 pixels (784 features when flattened)
- **Color:** Grayscale (pixel values 0-255)
- **Classes:** 10 (digits 0-9)
- **Working Subset:** 15,000 images (for practical training time)

## Approach

### Step 1: Data Loading
- Loaded MNIST using sklearn's fetch_openml
- Auto-downloads on first run, then uses cached version
- Each image stored as flat array of 784 pixel values

### Step 2: Exploratory Data Analysis
- Verified balanced class distribution across all digits
- Displayed random sample images to observe handwriting styles
- Created digit frequency bar chart
- Examined pixel-level intensity using heatmap visualization

### Step 3: Preprocessing
1. **Subsampling:** Selected 15,000 from 70,000 images
   - Full dataset makes SVM training extremely slow
   - Subset gives strong accuracy in reasonable time

2. **Normalization:** Divided pixels by 255 (range: 0-1)
   - Models train faster with smaller input values

3. **Train-Test Split:** 80% training / 20% testing
   - Stratified to keep digit proportions equal

4. **Standard Scaling:** Applied StandardScaler
   - Centers data at zero mean with unit variance
   - Important for SVM which uses distance calculations

### Step 4: Model Training

**Random Forest:** Ensemble of 150 decision trees. Each tree
trains on random data subset. Final answer = majority vote.
Naturally resistant to overfitting.

**SVM:** Finds optimal boundary between digit classes using
RBF kernel. Handles non-linear patterns by mapping data to
higher dimensions. Strong accuracy on image tasks.

### Step 5: Evaluation
- Accuracy scores for both models
- Per-digit precision, recall, and F1 scores
- Confusion matrices showing inter-digit confusions
- Identified commonly confused digit pairs

### Step 6: Visualization
- Sample predictions with green (correct) / red (wrong) labels
- Separate display of misclassified images for error analysis
- Many errors are genuinely ambiguous handwriting

## Results

| Model         | Accuracy |
|---------------|----------|
| Random Forest | ~95-96%  |
| SVM           | ~96-97%  |

**Best Model:** SVM

### Commonly Confused Digits
| Actual | Predicted As | Reason |
|--------|-------------|--------|
| 4 | 9 | Similar stroke with loop at top |
| 3 | 8 | Similar curved shape |
| 7 | 1 | Both vertical strokes |

## Output Files
| File | Description |
|------|-------------|
| sample_digits.png | Random images from dataset |
| digit_distribution.png | Samples per digit class |
| digit_detail.png | Pixel intensity heatmap |
| confusion_matrices.png | Both models evaluation |
| model_comparison.png | Accuracy comparison chart |
| sample_predictions.png | Correct and wrong predictions |
| wrong_predictions.png | Misclassified digits only |

## How to Run
```bash
pip install -r requirements.txt
python mnist_recognition.py

##Key Learnings
1. Image data needs normalization and scaling before training
2. Subsampling is practical when full dataset is too slow
3. SVM with RBF kernel works excellently for image classification
4. Random Forest gives strong results with less tuning
5. Confusion matrices reveal which digits confuse the model
6. Some errors are unavoidable due to genuinely bad handwriting

##Tools Used
Python, numpy, scikit-learn, matplotlib, seaborn
