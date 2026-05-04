# 🧠 Applied Artificial Neural Networks: Understanding Weight Initialization

## 1. Introduction
Developing Artificial Neural Networks (ANNs) requires more than just stacking layers. A critical but often overlooked challenge is **how to initialize the initial weights** of neurons. Poor weight initialization can lead to vanishing or exploding gradients, which prevents the network from learning effectively. 

This project explores a fundamental ANN applied to a "U-shape" non-linear dataset. It demonstrates how different neural network techniques—specifically **He Normal Initialization** and **Xavier Initialization** methodologies—affect model convergence and decision boundary formation.

**Real-world applications:** 
Proper weight initialization is the foundation for training deep networks in computer vision (CNNs), natural language processing (Transformers), and reinforcement learning. Without appropriate initialization, deep learning models cannot converge smoothly.

---

## 2. Dataset Understanding
For this experiment, we use the `ushape.csv` dataset.
- **Features (X_1, X_2):** Represent the 2D spatial coordinates of data points.
- **Target Variable (y):** A binary classification label (`0` or `1`).
- **Basic Insights:** The data is distributed in a non-linear, U-shaped scatter plot. A simple linear classifier cannot separate these classes effectively, making it an excellent candidate for a multi-layered ANN.

---

## 3. Data Preprocessing (Crucial Foundation)
Clean and scaled data ensures that the loss space is manageable and the gradients flow smoothly. 

### Feature Separation
We split the data features (X) from the target class (y).
*   **X** contains dimensions (X_1, X_2)
*   **y** contains binary classification targets.

### Scaling & Standardization (Contextual Overview)
Although not strictly applied to purely synthetic bounds in this mini-experiment, real-world data requires feature scaling.
- **Why scaling is needed in ANN:** Neural networks process inputs through weighted sums. If feature scales vary wildly (e.g., $X_1 \in [0, 1]$ and $X_2 \in [0, 1000]$), the gradient descent landscape becomes highly skewed (elliptical), making convergence painfully slow or unstable.
- **Standardization (Z-score):** $z = \frac{x - \mu}{\sigma}$ forces the data to have a mean of 0 and standard deviation of 1. It aligns well with the default operational ranges of activation functions like Sigmoid and Tanh, avoiding saturation zones out-of-the-box.

---

## 4. Artificial Neural Network (Core Section)

### Neuron Intuition
A single neuron applies a linear transformation followed by a non-linear activation.
1.  **Weighted Sum:** $z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$
2.  **Activation:** $a = f(z)$

### Activation Functions
- **ReLU (Rectified Linear Unit):** Used in the hidden layers. $f(z) = \max(0, z)$.
  *Why preferred?* It solves the vanishing gradient problem inherent to Sigmoid functions in deep networks and induces sparsity.
- **Sigmoid:** Used in the final output layer. $f(z) = \frac{1}{1 + e^{-z}}$.
  *Why preferred?* It accurately crushes our output to a $[0, 1]$ range, giving us our final probability for binary classification.

### Loss Function
To evaluate how far the predictions are from reality, we use **Binary Crossentropy (Log Loss)**.
- **Formula:** $L = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$
- **Intuition:** It penalizes the model heavily when it confidently predicts the wrong class.

### Backpropagation & Gradient Descent
Weights are iteratively adjusted by calculating the loss gradient with respect to each parameter. Gradients are propagated backward from the output layer to the input.

---

## 5. Model Architecture
- **Input Dimension:** 2
- **Hidden Layers:** 4 layers with 10 units each (`Dense(10)`).
- **Output Layer:** 1 unit.
- **Trade-offs:** Deeper and wider networks can learn more complex decision bounds (handling the U-shape easily) but risk overfitting and take longer to train. The `10x10x10x10` configuration is deep enough to necessitate good initialization logic.

---

## 6. Weight Initialization (The Central Theme)
### Why Custom Initialization?
If we start with $Weights = 0$, neurons learn symmetrically resulting in identical representations. If weights are too large, inputs explode; if too small, gradients vanish.

The notebook uses customized variance constraints (akin to **Xavier / Glorot Normal** initialization):
- **Variance constraint:** $W \sim \mathcal{N}(0, \frac{1}{\text{fan\_in}})$ 
- **Method in Code:** Explicit matrix overwriting via `np.random.randn(rows, cols) * np.sqrt(1 / fan_in)`.

---

## 7. Optimizers
We optimize using **Adam (Adaptive Moment Estimation)**.
- **Why preferred?** Unlike regular Stochastic Gradient Descent (SGD) which uses a single learning rate, Adam dynamically adapts the learning rates of each parameter by tracking first-order (momentum) and second-order (variance) gradient histories.

---

## 8. Model Training & Evaluation
We build the model with the following methodology:
- **Epochs (100):** Allowing the network 100 complete passes over the dataset.
- **Validation Split (0.2):** Setting aside 20% of the data to ensure the model generalizes rather than memorizes.
- **Result Output:** Mapped beautifully using `mlxtend`'s Decision Region charting tool mapping out the U-shaped boundary.

---

## 9. Insights and How to Run
### What Worked?
- Deep non-linear architectures with localized decision boundary tracking can comfortably solve U-shaped logical classifications.
- Overlapping weights based on the normal distribution and correctly sized variances ($* \sqrt{1/\text{fan\_in}}$) maintains active gradient flows compared to tiny $0.01$ multipliers (which cause vanishing gradients).

### Alternative Approaches
- **Logistic Regression:** Would completely fail here because the decision boundary is strictly linear.
- **Decision Trees:** Could solve it, but yields strict rectangular boundaries. ANN gives a smooth, continuous probabilistic boundary.

### How to Run
1. Install requirements: `pip install tensorflow numpy pandas matplotlib mlxtend`
2. Run Jupyter or VS Code Notebook: `weight_intz.ipynb`.