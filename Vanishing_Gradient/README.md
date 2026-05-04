# Vanishing Gradient: Deep Sigmoid vs ReLU on make_moons

## 1. Introduction
This notebook demonstrates the vanishing gradient problem by training a deep sigmoid network on a simple non-linear dataset. You will see that the loss can stall and early-layer weights barely update. Then you will apply fixes such as ReLU activations and shallower depth to restore learning.

Why this matters:
- Vanishing gradients are a core failure mode of deep networks.
- Understanding it helps you choose activations, architectures, and training tricks that actually learn.

Real-world significance:
- Deep models in vision, NLP, and speech can fail to train without proper design choices.
- This problem explains why modern networks rely on ReLU, BatchNorm, and residual connections.

## 2. Dataset Understanding
**Dataset:** synthetic `make_moons` from scikit-learn  
**Features:** 2D points (x1, x2)  
**Target:** binary class (0 or 1)

Why this dataset:
- Non-linear but low-dimensional, so we can visualize the task clearly.
- Perfect for isolating optimization behavior.

## 3. Data Preprocessing
- A simple train/test split is used.
- Feature scaling is not required here but can help if needed.

### Train-Test Split
We keep a test set to verify that any improvements are not just overfitting.

## 4. Artificial Neural Network (Core Section)

### Neuron Intuition
Each neuron computes:
$$z = w_1 x_1 + w_2 x_2 + b$$
$$a = f(z)$$

### Why Sigmoid Causes Vanishing Gradients
Sigmoid saturates for large positive or negative inputs:
$$\sigma(z) = \\frac{1}{1+e^{-z}}$$  
Derivative:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \\le 0.25$$

Across many layers, gradients multiply:
$$\\frac{\\partial L}{\\partial W_1} \\propto \\prod_{l=1}^{L} \\sigma'(z_l)$$  
Since each term is at most 0.25, the product rapidly shrinks toward 0.

## 5. Model Architecture
### Deep Sigmoid Network
- Many hidden layers with sigmoid activations.
- Purposefully designed to trigger vanishing gradients.

### ReLU Network (Fix)
- Same depth but ReLU activation.
- ReLU does not saturate for positive values, so gradients survive deeper.

## 6. Optimization Diagnostics
We compare:
- Loss curves (flat or slow decrease indicates stalled learning).
- Early-layer weight updates (small deltas indicate vanishing gradients).

Note: `old_weights - new_weights` is an update proxy, not a raw gradient in Adam.

## 7. Remedies for Vanishing Gradients
- **ReLU / Leaky ReLU:** avoids saturation.
- **Shallow models:** reduce gradient chain length.
- **He initialization:** keeps variance stable for ReLU.
- **BatchNorm:** stabilizes activations and gradients.
- **Residual connections:** allow gradients to bypass deep stacks.

## 8. Results & Insights
- Deep sigmoid often shows slow or stalled loss reduction.
- ReLU networks show healthier optimization and larger updates.

## 9. Alternative Approaches
- Logistic regression: simple baseline for shallow tasks.
- Decision trees: can solve moons but lack gradient-based insight.
- SVM with kernel: strong performance but not neural training.

## 10. How to Run
1. Install dependencies:
   - `pip install numpy matplotlib scikit-learn tensorflow keras`
2. Open the notebook and run cells in order.
