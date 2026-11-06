# Skin Cancer Detection Using Deep Learning

## Project Overview

This project focuses on detecting and classifying skin cancer lesions using deep learning techniques. It uses the HAM10000 (Human Against Machine with 10,000 training images) dataset to develop and evaluate multiple deep learning models for skin lesion classification.

The application integrates three trained models to support skin cancer prediction:

- **VGG16** – A convolutional neural network architecture for image classification.
- **ResNet** – A residual neural network used for feature extraction and classification.
- **Capsule Network (CapsNet)** – A deep learning architecture designed to capture spatial relationships between image features.

## Models Used

| Model | Saved Model |
|---|---|
| VGG16 | `vgg_functional.keras` |
| ResNet | `resnet_final.keras` |
| Capsule Network | `caps_model.keras` |

All three trained models are included in this repository.

## Dataset

**HAM10000 Dataset**

The project uses the HAM10000 dataset, which contains dermatoscopic images of pigmented skin lesions across seven diagnostic categories.

The dataset is not included in this repository due to its size.

Dataset source: [HAM10000 – Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Technologies Used

- Python
- TensorFlow
- Keras
- Deep Learning
- Convolutional Neural Networks (CNN)
- VGG16
- ResNet
- Capsule Networks
- Jupyter Notebook

## Project Structure

```text
skin-cancer-detection/
│
├── app.py
├── caps_model.keras
├── resnet_final.keras
├── vgg_functional.keras
├── .gitignore
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/Hidha09/skin-cancer-detection.git
```

2. Navigate to the project directory:

```bash
cd skin-cancer-detection
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

5. Install the required dependencies:

```bash
pip install tensorflow numpy pandas pillow
```

Install any additional dependencies required by the application.

6. Run the application:

```bash
python app.py
```

## Key Features

- Skin lesion image classification using deep learning.
- Integration of three trained models: VGG16, ResNet and CapsNet.
- Saved Keras models for inference.
- Application-based prediction interface.

## Future Improvements

- Improve model performance through further experimentation.
- Explore additional deep learning architectures.
- Enhance prediction accuracy and generalisation.
- Develop a more comprehensive user interface.

## Disclaimer

This project is developed for educational and research purposes. It is not intended to replace professional medical diagnosis or clinical evaluation.

## Author

**Hidha C H**

Integrated M.Sc. Computer Science – Artificial Intelligence & Machine Learning

[GitHub Profile](https://github.com/Hidha09)
