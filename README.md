# Microsoft ML - Deep Learning

Neural network classification model for wheat seeds species.

- Dataset: seeds.csv (Kama, Rosa, Canadian wheat)
- Model: MLPClassifier (3 hidden layers)
- Metrics: classification report + confusion matrix
- Saved model: mlp_model.pkl

---

# Microsoft Machine Learning - Deep Learning Module

This repository contains the work from Microsoft's **"Train and evaluate deep learning models"** module.

## What I Learned

- How **artificial neural networks** mimic the human brain using neurons, weights, bias, and activation functions.
- Core concepts of **Deep Neural Networks (DNN)**: input layer, hidden layers, output layer, fully connected networks.
- Training process: **epochs**, **forward pass**, **loss calculation**, **backpropagation**, and **optimizers** (SGD, Adam, etc.).
- Importance of **learning rate** and how it affects model convergence.
- **Convolutional Neural Networks (CNN)** basics:
  - Convolution layers + filters/kernels
  - Pooling layers (max pooling)
  - Dropout layers (to prevent overfitting)
  - Flattening and fully connected layers
- **Transfer Learning** concept – reusing pre-trained feature extraction layers.

## Project in this Repo

- **Dataset**: `seeds.csv` – wheat seed measurements (Kama, Rosa, Canadian species)
- **Model**: Multi-Layer Perceptron (`MLPClassifier`) with 3 hidden layers (30 neurons each)
- **Task**: Multiclass classification of wheat seeds
- **Metrics**: Classification report + Confusion Matrix
- **Saved model**: `mlp_model.pkl`

## Files
- `seeds.csv` – raw dataset
- `script.py` – training + evaluation script
- `mlp_model.pkl` – trained model
- `deep-learning.pysb` – original notebook content

## Technologies Used
- Python, pandas, scikit-learn
- MLPClassifier (neural network)

---

**Status**: Completed Microsoft ML Deep Learning module.