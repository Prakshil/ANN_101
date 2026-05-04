# Dropout Regularization in Artificial Neural Networks

## Overview
This folder contains the `dropout.ipynb` notebook which explores the concept of **Dropout**, one of the most effective ways to prevent Neural Networks from overfitting on small or noisy datasets.

## Project Structure
- `dropout.ipynb`: The notebook visually comparing a standard Deep Neural Network against one implementing Dropout layers.

## Key Concepts Demonstrated
1. **Overfitting**: Creating an intentionally wide model and training it until it perfectly fits random noise points rather than the true underlying distribution.
2. **Dropout Mechanism**: Randomly dropping a percentage (e.g. 20%) of neurons during training passes. This forces the remaining neurons to become more robust and learn generalized features rather than combining to memorize exact data coordinates.
3. **Visual Comparison**: Uses matplotlib to overlay the predictions, clearly showcasing an overfitted jagged regression line vs a smoothed, generalized curve achieved via dropout.

## Instructions
1. Ensure dependencies are installed: `pip install numpy pandas tensorflow sklearn matplotlib`
2. Open `dropout.ipynb`.
3. Execute the cells chronologically. The custom regression data is generated synthetically at runtime.