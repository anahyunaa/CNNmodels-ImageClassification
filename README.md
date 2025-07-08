# Waste Classification using Deep Learning

A deep learning project for waste classification using transfer learning with pre-trained CNN models (VGG19, ResNet50, and EfficientNetB0).

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Features](#features)
- [Team](#team)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a waste classification system using deep learning techniques. The goal is to automatically classify waste images into 6 different categories: cardboard, glass, metal, paper, plastic, and trash. The project compares the performance of three popular pre-trained CNN architectures through transfer learning.

## 📊 Dataset

The project uses the TrashNet dataset with the following categories:
- **Cardboard** - Cardboard waste items
- **Glass** - Glass bottles and containers
- **Metal** - Metal cans and containers
- **Paper** - Paper waste items
- **Plastic** - Plastic bottles and containers
- **Trash** - General trash items

### Dataset Split:
- **Training**: 80% of the data
- **Validation**: ~13% of the data
- **Test**: ~7% of the data

All images are resized to 224x224 pixels for model input.

## 📁 Project Structure

```
tubes_fix/
│
├── test.ipynb                          # Main Jupyter notebook
├── README.md                           # Project documentation
│
├── DatasetTrashnet/
│   └── dataset-resized/               # Original dataset
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
│
├── data_split/                        # Processed dataset splits
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/                            # Saved model files
│   ├── VGG19_fine_tuned_final.keras
│   ├── ResNet50_fine_tuned_final.keras
│   ├── EfficientNetB0_fine_tuned_final.keras
│   └── efficientnetb0_notop.h5
│
└── logs/                              # TensorBoard logs
    └── fit/
        ├── VGG19_initial/
        ├── VGG19_finetune/
        ├── ResNet50_initial/
        ├── ResNet50_finetune/
        ├── EfficientNetB0_initial/
        └── EfficientNetB0_finetune/
```

## 🤖 Models

This project implements and compares three state-of-the-art CNN architectures:

### 1. VGG19
- **Architecture**: 19-layer Visual Geometry Group network
- **Pre-trained on**: ImageNet
- **Fine-tuning**: Last 4 layers unfrozen

### 2. ResNet50
- **Architecture**: 50-layer Residual Network
- **Pre-trained on**: ImageNet
- **Fine-tuning**: Last 22 layers unfrozen

### 3. EfficientNetB0
- **Architecture**: Efficient Neural Network B0
- **Pre-trained weights**: Custom weights (efficientnetb0_notop.h5)
- **Fine-tuning**: Last 20 layers unfrozen

### Training Strategy

Each model follows a two-phase training approach:

1. **Initial Training Phase**:
   - Frozen base model layers
   - Learning rate: 1e-3
   - Epochs: Up to 80 (with early stopping)

2. **Fine-tuning Phase**:
   - Unfrozen top layers
   - Learning rate: 1e-5
   - Epochs: Up to 20 (with early stopping)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Required Packages

```bash
pip install tensorflow
pip install numpy pandas matplotlib seaborn
pip install opencv-python pillow
pip install scikit-learn
pip install pathlib
```

### Alternative: Using conda

```bash
conda install tensorflow-gpu
conda install numpy pandas matplotlib seaborn
conda install opencv pillow
conda install scikit-learn
```

## 💻 Usage

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd tubes_fix
   ```

2. **Prepare the dataset**:
   - Place your dataset in `DatasetTrashnet/dataset-resized/` folder
   - Ensure each category has its own subfolder

3. **Run the notebook**:
   ```bash
   jupyter notebook test.ipynb
   ```

4. **Execute cells sequentially**:
   - **EDA Section**: Analyze dataset distribution and image properties
   - **Preprocessing**: Split data and create data loaders
   - **Model Training**: Train each model (VGG19, ResNet50, EfficientNetB0)
   - **Evaluation**: Compare model performances

## 📈 Results

The project provides comprehensive evaluation metrics:

- **Test Accuracy**: Final accuracy on unseen test data
- **Classification Report**: Precision, recall, and F1-score for each class
- **Confusion Matrix**: Visual representation of classification performance
- **Training History**: Accuracy and loss curves during training

### Key Features of Results:
- Comparison table of all three models
- Visual performance comparison charts
- Detailed per-class performance metrics
- Training history visualization

## ✨ Features

### Data Analysis
- **Dataset Distribution**: Visual analysis of class balance
- **Image Properties**: Analysis of image dimensions, aspect ratios, and blur metrics
- **Sample Visualization**: Display sample images from each category

### Data Preprocessing
- **Automatic Data Splitting**: Stratified train/validation/test split
- **Data Augmentation**: Random flip, rotation, and zoom
- **Class Balancing**: Automatic class weight computation
- **Batch Processing**: Efficient data loading with TensorFlow datasets

### Model Training
- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Two-Phase Training**: Initial training + fine-tuning approach
- **Callbacks**: Model checkpointing, early stopping, and TensorBoard logging
- **Performance Monitoring**: Real-time training progress tracking

### Evaluation
- **Multiple Metrics**: Accuracy, precision, recall, F1-score
- **Visual Analysis**: Confusion matrices and performance charts
- **Model Comparison**: Side-by-side performance comparison

## 👥 Team

**Kelompok Bebas**

This project was developed by:

| Name | Student ID | Email | Role |
|------|------------|-------|------|
| Fairus Rajendra Wiranata | 101121050 | fairus344@gmail.com | Team Leader / Model Architecture |
| Ni Putu Merta Bhuana N | 105222008 | putumertabhuana@gmail.com | Data Preprocessing / EDA |
| Arshanda Geulis Nawajaputri | 105222045 | arshandagn06@gmail.com | Model Training / Optimization |
| Yan Stephen Christian Immanuel | 105222010 | yan.stephen49@gmail.com | Evaluation / Documentation |

### Contributions:
- **Fairus Rajendra Wiranata**: Designed model architectures, implemented transfer learning approach
- **Ni Putu Merta Bhuana N**: Performed exploratory data analysis, data preprocessing and augmentation  
- **Arshanda Geulis Nawajaputri**: Handled model training, hyperparameter tuning, and performance optimization
- **Yan Stephen Christian Immanuel**: Conducted model evaluation, created visualizations, and wrote documentation

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes

- The project uses TensorFlow/Keras for deep learning implementation
- All models are saved in `.keras` format for compatibility
- TensorBoard logs are generated for training visualization
- The notebook includes error handling for missing files and directories

## 🏆 Performance Tips

- Use GPU acceleration for faster training
- Monitor TensorBoard logs for training insights
- Adjust batch size based on available memory
- Consider data augmentation parameters based on your dataset

---

**Team**: Kelompok Bebas  
**Members**: Fairus Rajendra Wiranata, Ni Putu Merta Bhuana N, Arshanda Geulis Nawajaputri, Yan Stephen Christian Immanuel  
**Course**: Machine Learning  
**Institution**: Universitas Pertamina
**Date**: July 2025
