# MNIST Digit Classification with ANN

## 1. Introduction
This project classifies handwritten digits (0-9) using a simple fully connected ANN. It demonstrates how image data can be flattened and fed into dense layers for multi-class classification.

Why ANN here:
- Shows the baseline performance of a dense network on image data
- Serves as a comparison point for CNNs

Real-world applications:
- Document digitization
- OCR pipelines and handwriting recognition

## 2. Dataset Understanding
**Dataset:** MNIST (Keras built-in)

**Features (inputs):**
- 28x28 grayscale pixel values (0-255)

**Target:**
- Digit class (0 to 9)

**Data types:**
- Numeric arrays for images, integer labels

**Basic insights:**
- Pixel intensities are bounded, making min-max scaling natural
- Each image has 784 features after flattening

**Potential issues:**
- Dense networks ignore spatial structure
- Model can overfit if capacity is high

## 3. Data Preprocessing
### Normalization / Standardization
**Min-Max scaling:**
$$x' = \frac{x}{255}$$
Rescales pixel values to [0, 1] to stabilize gradients.

### Train-Test Split
MNIST already provides a standard train/test split to compare results.

## 4. Artificial Neural Network (Core Section)
### Neuron Intuition
$$z = w_1 x_1 + \dots + w_n x_n + b$$
$$a = f(z)$$

### Activation Functions
- **ReLU** in hidden layers for nonlinearity
- **Softmax** in output layer for class probabilities

### Loss Function
**Sparse Categorical Crossentropy:**
$$L = -\log(\hat{y}_{true})$$
This form uses integer labels instead of one-hot vectors.

## 5. Model Architecture
- Flatten layer to convert 28x28 to 784
- Dense(128, ReLU) for feature extraction
- Dense(10, Softmax) for classification

Trade-offs:
- Simple architecture is easy to interpret
- CNNs typically outperform ANNs on images

## 6. Dropout (Very Important)
Dropout can reduce overfitting by randomly disabling neurons during training. Not used here, but often improves dense networks on MNIST.

## 7. Optimizers
Adam is used for fast convergence and stable training.

## 8. Model Training
- Epochs define passes over the data
- Validation split monitors generalization

## 9. Hyperparameter Tuning (If Used)
Potential tuning targets:
- Hidden units (64, 128, 256)
- Learning rate
- Dropout rate

## 10. Evaluation
- Accuracy on the test set
- Confusion matrix for error analysis

## 11. Results and Insights
- Baseline ANN provides decent accuracy
- Confusions often occur between visually similar digits

## 12. Alternative Approaches
- **CNNs:** use spatial structure and typically outperform ANNs
- **SVM:** good for small feature sets but less scalable for images
- **KNN:** simple but slower at inference

## 13. How to Run
1. Install dependencies:
   - `pip install numpy tensorflow scikit-learn matplotlib seaborn`
2. Open [MNIST_ANN.ipynb](MNIST_ANN.ipynb)
3. Run cells in order to train and evaluate the model
