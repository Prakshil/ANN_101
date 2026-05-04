# Early Stopping in Artificial Neural Networks

## Overview
This folder contains the `early_stopping.ipynb` notebook which provides a visual deep-dive into utilizing Keras Callbacks—specifically `EarlyStopping`—to halt training before a model begins to overfit.

## Project Structure
- `early_stopping.ipynb`: Contains code to synthesize non-linear data, plot decision boundaries, and track training via metrics.

## Key Concepts Demonstrated
1. **The Overfitting Horizon**: Training a 256-neuron network on a small, noisy circular dataset for 3500 epochs to force extreme overfitting.
2. **Loss Divergence**: Graphed training loss hitting zero while validation loss spikes into a U-shape.
3. **EarlyStopping Callback**: Setting up monitoring for `val_loss`, waiting for a configurable `patience`, and automatically stopping training when learning stops generalizing.
4. **Decision Boundary Smoothness**: Visualizing how an early-stopped model produces a smooth, robust separating boundary whereas a 3500-epoch model generates bizarre, disjointed blobs to classify noise.

## Instructions
1. Dependencies required: `pip install mlxtend numpy pandas tensorflow sklearn seaborn matplotlib`
2. Open `early_stopping.ipynb`.
3. Follow the sequence. The first half purposefully ruins a model, while the second half uses Early Stopping to automate the rescue.