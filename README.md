# Anomaly Detection with Contrastive Learning and GAN

This project implements an anomaly detection framework for multivariate time series data, integrating a Transformer-based Autoencoder with contrastive learning and Generative Adversarial Networks (GAN). The goal is to improve anomaly detection by addressing overfitting and expanding the training dataset with data augmentation techniques.

## Features
- **Transformer-Based Autoencoder**: Compresses and reconstructs time series data, detecting anomalies via reconstruction errors.
- **Contrastive Learning**: Uses a contrastive loss function to encourage similar samples to be closer and dissimilar samples to be farther apart.
- **Generative Adversarial Networks (GAN)**: Generates synthetic data to augment the dataset and reduce overfitting.
- **Data Augmentation**: Utilizes geometric distribution masks to augment the training data.

## Datasets
The project uses the following datasets for experimentation and validation:
- **Credit Card Fraud Detection Dataset**: Contains credit card transactions with labeled fraud cases.
- Additional datasets can be added based on project needs.

## Installation
To run the project, you'll need Python and the required packages. Install them with:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
