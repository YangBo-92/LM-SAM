# LM-SAM: Efficient Transfer of Segment Anything Model for Medical Image Segmentation via Structural Prior and Detail Preservation

Official PyTorch implementation of **"LM-SAM: Efficient Transfer of Segment Anything Model for Medical Image Segmentation via Structural Prior and Detail Preservation"**.

## 🛠️ Installation

### Requirements

```
Python = 3.10
PyTorch = 2.6.0
```

### Dependencies List

Create a `requirements.txt` file with:

```
numpy>=2.2.6
opencv-python>=4.12.0.88
scikit-learn>=1.7.1
torch>=2.6.0
torchvision>=0.21.0
```

## 📁 Project Structure

```
LM-SAM/
├── SAM/                          # SAM model implementation
├── model_utils/                  # Utility modules
│   ├── cfg.py                   # Configuration file
│   ├── dataset_split.py         # Dataset handling
│   ├── evalution_segmentation.py # Evaluation metrics
│   └── class_dict.csv            # Label mapping table
├── dataset/                      # Dataset directory
├── weight/                       # Model checkpoints directory
├── train_SAM.py                 # Main training script
├── test_SAM.py                  # Testing script  
├── predict_SAM.py               # Prediction script
└── README.md                    # This file
```

## 📊 Supported Datasets

| Dataset      | Description | Images | Modality | Download Link |
|--------------|-------------|--------|----------|---------------|
| **BUSI**     | Breast Ultrasound Segmentation | 780 | Ultrasound | [Link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| **ISIC2018** | Skin Lesion Segmentation | 2,594 | Dermoscopy | [Link](https://challenge2018.isic-archive.com/) |
| **Kvasir**   | Polyp Segmentation | 1,000 | Endoscopy | [Link](https://datasets.simula.no/kvasir-seg/) |
| **JSRT**     | Chest X-ray Segmentation | 247 | X-ray | [Link](http://db.jsrt.or.jp/eng.php) |
| **LiTS**     | Liver Tumor Segmentation Challenge | 131 | CT | [Link](https://competitions.codalab.org/competitions/17094#participate) |


### Dataset Preparation

1. **Download datasets** from the links above
2. **Organize your data** in the following structure:

```
dataset/
├── DATASET_NAME/
│   ├── images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── masks/
│       ├── image001.png
│       ├── image002.png
│       └── ...
```

3. **Update configuration** in `model_utils/cfg.py`:

```python
# Dataset paths
TRAIN_ROOT = "./dataset/YOUR_DATASET/images"
TRAIN_LABEL = "./dataset/YOUR_DATASET/masks"

# Training parameters
BATCH_SIZE = 32
EPOCH_NUMBER = 200
lr = 0.0001
image_size = 256
```
## 📦 Model Weights

Pre-trained LM‑SAM checkpoints can be downloaded from Baidu Netdisk:

**🔗 Baidu Netdisk Download**  
**Link:** https://pan.baidu.com/s/1Uw4OVaK1oO1OUSvq9UxuZw  
**Extraction code:** 2tmq

```
Download and place them under `weight/` directory:
```

## 🚀 Quick Start

### 1. Training

```bash
# Train on default dataset (configured in cfg.py)
python train_SAM.py

# Monitor training progress
# Check the console output for loss and metrics
# Model checkpoints will be saved in ./weight/ directory
```

### 2. Testing

```bash
# Test the trained model
python test_SAM.py

# Results will be displayed in console
```

### 3. Prediction

```bash
python predict_SAM.py
```

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [Awesome-Parameter-Efficient-Transfer-Learning Public](https://github.com/facebookresearch/segment-anything)
- PyTorch and open-source deep learning community
