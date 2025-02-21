# 🚀 AI Examples Repository

Welcome to the **AI Examples Repository**! This repository contains various **machine learning** and **deep learning** examples using **Scikit-Learn** and **TensorFlow**.

## 📌 About
This repository includes multiple AI-related examples, covering topics like:
✅ **Supervised Learning** (Classification, Regression)
✅ **Neural Networks (Deep Learning)**
✅ **Convolutional Neural Networks (CNNs)**
✅ **Model Training & Prediction**

All examples are implemented in **Python 3.11** and demonstrate how to build and train machine learning models efficiently.

---

## 📂 Examples Included
### 1️⃣ **Logistic Regression with Scikit-Learn**
- Implements **classification** using **Logistic Regression**.
- Uses **Iris Dataset** from `sklearn.datasets`.
- Demonstrates **data preprocessing, model training, evaluation, and predictions**.

### 2️⃣ **CNN for Image Classification with TensorFlow**
- Builds a **Convolutional Neural Network (CNN)** using `tensorflow.keras`.
- Trains on the **CIFAR-10 dataset** to classify images into **10 categories**.
- Includes **data normalization, model architecture, training, and custom image prediction**.
- Supports **saving and loading models** to avoid retraining every time.

### 3️⃣ **Transformer Model for Sentiment Analysis with TensorFlow**
- Implements a **Transformer-based neural network** using `tensorflow.keras`.
- Trains on the **IMDB Movie Reviews Dataset** for binary sentiment classification (**positive** or **negative**).
- Includes:
  - **Data downloading** and **preprocessing**.
  - Custom **Transformer architecture** with positional embeddings and multi-head attention.
  - Functions for **model training**, **prediction**, and **evaluation**.
---

## ⚙️ Installation & Setup
Before running the code, install the required dependencies:

```sh
pip install numpy pandas matplotlib scikit-learn tensorflow opencv-python pillow
```

Ensure that you're using **Python 3.11**:

```sh
python --version
```

---

## 🚀 Running the Examples
### **Run each of the examples individually and see the result.**
```sh
python linear_regression.py
```

---

## 📌 Saving & Loading Models
To avoid re-training the model every time, the "cnn_image_classification_custom_image.py" script automatically **saves** the trained model as:
```
cifar10_model.h5
```
If the model is already trained, it will be **loaded automatically** instead of re-training.

---

## 🎯 Future Improvements
- Adding **Object Detection models**.
- Implementing **Reinforcement Learning** algorithms.

---

## 📝 License
This project is open-source under the **MIT License**. Feel free to use, modify, and contribute! 🚀

---

🔗 **Stay Connected**
For updates and discussions, feel free to **follow the repository** or **open an issue** if you have any questions!
