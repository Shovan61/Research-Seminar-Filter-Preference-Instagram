# Instagram Filter Preference Prediction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project that predicts which Instagram filter a user would prefer for a given image, comparing three different CNN architectures (ResNet50, MobileNetV2, and EfficientNetB0).

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

---

## 🎯 Project Overview

### Problem Statement
Predict which Instagram filter a user would prefer for a given image. Given an input image (either AI-generated or real photograph), the model outputs a probability distribution over 19 possible Instagram filters.

### Objectives
- Train and compare three different deep learning models for filter classification
- Evaluate model performance using accuracy, classification reports, and confusion matrices
- Understand visual features that influence filter preferences

### Applications
- Automated photo editing recommendations
- User preference modeling for social media platforms
- Aesthetic preference analysis across AI vs. real images

---

## 📊 Dataset

### Data Structure

data_Instagram/
├── prefs.csv # Mapping file (image → preferred filter)
├── ai/ # AI-generated images (54 images)
│ ├── p01.png
│ ├── p02.png
│ └── ...
├── real/ # Real photographs (54 images)
│ ├── p01.png
│ ├── p02.png
│ └── ...
├── ai_pref/ # AI images with preferred filters applied
└── real_pref/ # Real images with preferred filters applied

text

### CSV Structure (prefs.csv)
| Column | Description | Example |
|--------|-------------|---------|
| `img` | Path to image | `ai/p01.png` |
| `pid` | Participant/Image ID | `p01` |
| `filter` | Preferred filter name | `mayfair` |

### Filter Classes (19 total)
aden, brannan, brooklyn, clarendon, earlybird, gingham, hudson,
inkwell, kelvin, lark, lofi, mayfair, org, perpetua, rise,
slumber, valencia, walden, xpro2

text

### Data Statistics
| Split | Count |
|-------|-------|
| Total images | 108 |
| AI images | 54 |
| Real images | 54 |
| Training set (80%) | 86 |
| Testing set (20%) | 22 |

---

## 🧠 Theoretical Background

### Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks for image data that preserve spatial relationships through convolution operations.
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Input │────▶│ Convolution │────▶│ Pooling │────▶│ Dense │
│ Image │ │ Layer │ │ Layer │ │ Layers │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
│ │ │
▼ ▼ ▼
Detects features Reduces dimensions Makes final
(edges, colors) (speed, memory) classification

text

### Key Components

**1. Convolutional Layer**
- Applies filters (kernels) that slide across the image
- Detects patterns like edges, corners, and textures
- Output: Feature maps

**2. Activation Function (ReLU)**
ReLU(x) = max(0, x)

text
Introduces non-linearity and turns negative values to zero.

**3. Pooling Layer (Max Pooling)**
[1, 3] [2, 4] → [3, 4]

text
Reduces spatial dimensions, keeping only the most important features.

**4. Dense (Fully Connected) Layers**
- Final decision-making layers
- Combine features to make predictions

### Transfer Learning

Instead of training from scratch (requires millions of images), we use pre-trained models:
Pre-trained on ImageNet (14M images, 1000 classes)
↓
Feature Extractor
↓
New Classification Head (19 filters)
↓
Fine-tuned on our dataset

text

**Benefits:**
- Requires less training data
- Faster convergence
- Better performance (leverages existing visual knowledge)

---

## 🏗️ Model Architectures

### Model 1: ResNet50
- **Year:** 2015
- **Parameters:** 25 million
- **Key Feature:** Skip connections (residual blocks) that prevent vanishing gradients
Input (224×224×3)
↓
ResNet50 Base (frozen)
├── 49 convolutional layers
├── Skip connections
└── Pre-trained on ImageNet
↓
GlobalAveragePooling2D
↓
Dense(512, ReLU)
↓
Dropout(0.5)
↓
Dense(256, ReLU)
↓
Dropout(0.3)
↓
Dense(N_classes, Softmax)

text

**Skip Connection Diagram:**
Input → Conv → ReLU → Conv → Add → Output
↓_________________↑

text

### Model 2: MobileNetV2
- **Year:** 2018
- **Parameters:** 3.5 million (much smaller!)
- **Key Feature:** Depthwise separable convolutions for efficiency
Input (224×224×3)
↓
MobileNetV2 Base (frozen)
├── Depthwise separable convolutions
├── Inverted residuals
└── 3.5M parameters
↓
GlobalAveragePooling2D
↓
Dense(256, ReLU)
↓
Dropout(0.4)
↓
Dense(128, ReLU)
↓
Dropout(0.3)
↓
Dense(N_classes, Softmax)

text

**Depthwise Separable Convolution:**
Standard Conv: 3×3×3×32 = 864 parameters
Depthwise Separable: (3×3×3) + (1×1×3×32) = 27 + 96 = 123 parameters
(7x fewer parameters!)

text

### Model 3: EfficientNetB0
- **Year:** 2019
- **Parameters:** 5.3 million
- **Key Feature:** Compound scaling (scales depth, width, resolution together)
Input (224×224×3)
↓
EfficientNetB0 Base (frozen)
├── Compound scaling
├── MBConv blocks with squeeze-and-excitation
└── Best accuracy-to-parameter ratio
↓
GlobalAveragePooling2D
↓
Dense(512, ReLU)
↓
Dropout(0.5)
↓
Dense(256, ReLU)
↓
Dropout(0.3)
↓
Dense(N_classes, Softmax)

text

### Why Freeze the Base Model?
```python
base_model.trainable = False
Preserves pre-trained ImageNet knowledge while only training the new classification layers.

🚀 Installation
Prerequisites
Python 3.8+

Google Colab (recommended) or local GPU

Google Drive for dataset storage

Clone the Repository
bash
git clone https://github.com/yourusername/instagram-filter-prediction.git
cd instagram-filter-prediction
Install Dependencies
bash
pip install tensorflow pandas numpy opencv-python matplotlib seaborn scikit-learn
Dataset Setup
Upload your dataset to Google Drive at: /content/drive/MyDrive/data_Instagram/

Ensure the following structure:

text
data_Instagram/
├── prefs.csv
├── ai/
├── real/
├── ai_pref/
└── real_pref/
📖 Usage
Run in Google Colab
Open Google Colab

Upload the notebook

Change runtime type to GPU: Runtime → Change runtime type → T4 GPU

Update the DATASET_PATH variable

Run all cells

Key Code Sections
Load and preprocess data:

python
df = pd.read_csv('prefs.csv')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['filter'])
Create data generator with augmentation:

python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
Train a model:

python
model = create_resnet50(num_classes)
history = model.fit(train_generator, epochs=50, validation_data=test_generator)
Predict on new images:

python
predictions = predict_filter('path/to/image.jpg', model_efficientnet)
Configuration Parameters
python
IMG_SIZE = (224, 224)    # Input image size
BATCH_SIZE = 32          # Images per batch
EPOCHS = 50              # Training epochs (30-50 recommended)
TEST_SIZE = 0.2          # 20% testing, 80% training
📊 Results
Expected Performance (after 50 epochs)
Model	Validation Accuracy	Inference Speed	Model Size
ResNet50	~35%	Medium	95 MB
MobileNetV2	~30%	Fast	42 MB
EfficientNetB0	~40%	Medium	89 MB
*Note: Random chance baseline = ~9% (1/11 classes)*

Sample Confusion Matrix
text
Predicted →
True     brooklyn  clarendon  lofi  ...
↓
brooklyn     2         1        0
clarendon    1         3        0
lofi         0         0        2
Common Misclassifications
Filter Pair	Reason
lofi ↔ clarendon	Similar warm tones
brooklyn ↔ mayfair	Both vintage style
gingham ↔ valencia	Similar contrast levels
Sample Prediction Output
text
📊 EfficientNetB0 Predictions for p01.png:
   1. clarendon: 45.2%
   2. mayfair: 23.1%
   3. brooklyn: 12.5%
📁 Project Structure
text
instagram-filter-prediction/
│
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── instagram_filter_project.ipynb # Main Colab notebook
│
├── results/                      # Output directory
│   ├── sample_images.png
│   ├── training_history.png
│   ├── confusion_matrix_*.png
│   ├── model_comparison.png
│   ├── label_encoder.pkl
│   └── *_filter_model.h5        # Saved models
│
└── data/                        # Dataset (not included in repo)
    ├── prefs.csv
    ├── ai/
    ├── real/
    ├── ai_pref/
    └── real_pref/
🔧 Data Preprocessing Steps
1. Label Encoding
Converts filter names to integers:

text
'aden'      → 0
'brannan'   → 1
'brooklyn'  → 2
'clarendon' → 3
...
2. Image Rescaling
python
Pixel values: 0-255 → 0-1  # Divide by 255
3. Data Augmentation
Creates variations to prevent overfitting:

Rotation: ±20 degrees

Shifts: ±20% horizontal/vertical

Flip: Horizontal mirroring

Zoom: ±20%

4. Handling Rare Classes
Filters with only one sample are automatically removed to ensure valid train-test splitting.

📈 Evaluation Metrics
Accuracy
text
Accuracy = (Correct Predictions) / (Total Predictions) × 100%
Classification Report Metrics
Metric	Formula	Meaning
Precision	TP/(TP+FP)	Of predicted X, how many were actually X?
Recall	TP/(TP+FN)	Of actual X, how many were correctly found?
F1-Score	2×(P×R)/(P+R)	Harmonic mean of precision and recall
Confusion Matrix
Shows which classes are commonly confused with each other.

🚀 Future Improvements
Increase epochs to 50-100 for better accuracy

Fine-tune base models (unfreeze some layers)

Collect more data (current 108 images is limited)

Implement cross-validation for robust evaluation

Use ensemble methods to combine multiple models

Add Grad-CAM visualization to see what the model is learning

Deploy as web app using TensorFlow.js or FastAPI

📚 References
ResNet: He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.

Paper

MobileNetV2: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.

Paper

EfficientNet: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

Paper

Transfer Learning: Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. IEEE Transactions.

ImageNet: Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. CVPR.

👥 Authors
Shovan Mazumder

Audiovisual Technology Group, TU Ilmenau

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Prof. Dr.-Ing. Steve Göring for project guidance

Audiovisual Technology Group, TU Ilmenau

AVT Topic 2 course instructors

📧 Contact
For questions or feedback, please contact:

Email: your.email@tu-ilmenau.de

GitHub: @yourusername

📊 Badges
https://colab.research.google.com/assets/colab-badge.svg
https://img.shields.io/badge/Made%2520with-TensorFlow-FF6F00?logo=tensorflow
https://img.shields.io/badge/GPU-T4%2520%257C%2520V100-blue?logo=nvidia

Last Updated: May 2025

text

---

## How to Use This README:

1. **Copy the entire markdown code above**
2. Go to your GitHub repository
3. Click **"Add file"** → **"Create new file"**
4. Name it `README.md`
5. **Paste** the content
6. **Commit** the file

## Or create it locally:

```bash
# Create README.md file
nano README.md

# Paste the content, save and exit

# Push to GitHub
git add README.md
git commit -m "Add comprehensive README documentation"
git push origin main
This README includes:

✅ Project overview and objectives

✅ Dataset description and structure

✅ Theoretical background (CNNs, transfer learning)

✅ Detailed model architectures

✅ Installation and usage instructions

✅ Results and evaluation metrics

✅ Project structure

✅ References and citations

✅ Badges for visual appeal

The README is fully compatible with GitHub's markdown rendering and includes proper formatting, tables, code blocks, and emojis for better readability!
