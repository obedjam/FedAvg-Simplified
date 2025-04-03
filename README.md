Federated Learning with CNN on MNIST

Overview

This project implements a Federated Learning approach using a Convolutional Neural Network (CNN) on the MNIST dataset. The goal is to simulate decentralized training where multiple clients contribute to model training without sharing raw data.

Features

Implementation of a CNN for image classification.

Federated learning simulation across multiple clients.

Model training, aggregation, and evaluation.

Visualization of validation loss per client.

Requirements

Ensure you have the following dependencies installed:

pip install torch torchvision matplotlib numpy

Installation

Clone the repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo

How to Run

Run the script to start the training process:

python main.py

Make sure that the MNIST dataset is available or will be downloaded automatically to data/mnist/.

Project Structure

├── data/                   # Directory for MNIST dataset
├── main.py                 # Main script to run training
├── model.py                # CNN model definition
├── federated.py            # Federated learning implementation
├── utils.py                # Helper functions
├── README.md               # Project documentation

Results

The script prints the global model accuracy after each training round. The validation loss for each client is also plotted to observe training progress.
