# Hyperparameter Tuning with Keras Tuner

## 1. Introduction
This project demonstrates hyperparameter tuning for a simple neural network using Keras Tuner. Instead of choosing the number of units and learning rate manually, we let a search algorithm explore the space and select the best configuration.

Why tuning matters:
- Manual search is slow and biased
- Automated tuning often yields better accuracy with fewer experiments

Real-world applications:
- Model selection pipelines
- AutoML workflows
- Production model optimization

## 2. Dataset Understanding
This notebook assumes you already have:
- `x_train`: training features
- `y_train`: training targets

Any tabular regression dataset can be used. Ensure features are scaled and the target is numeric.

## 3. Data Preprocessing
### Normalization / Standardization
For dense networks, standardization is common:
$$z = \frac{x - \mu}{\sigma}$$

Scaling keeps gradients stable and makes the learning rate more predictable.

### Train-Test Split
A validation split inside the tuner search provides a fair comparison between hyperparameter trials.

## 4. Artificial Neural Network (Core Section)
### Neuron Intuition
$$z = w_1 x_1 + \dots + w_n x_n + b$$
$$a = f(z)$$

### Activation Functions
- **ReLU** for hidden layers
- **Linear** output for regression

### Loss Function
**Mean Squared Error:**
$$L = \frac{1}{n} \sum (y - \hat{y})^2$$

## 5. Model Architecture
The `build_model` function defines a simple network:
- Dense layer with tunable units (8 to 128)
- Linear output for regression

## 6. Dropout (Very Important)
Dropout is not included in this baseline, but can be added as another hyperparameter to reduce overfitting.

## 7. Optimizers
Adam is tuned by selecting the learning rate with a log-scale search.

## 8. Model Training
Each trial trains for a fixed number of epochs with a validation split.

## 9. Hyperparameter Tuning (Detailed)
### Keras Tuner
Keras Tuner automates trial selection and evaluation.

**Why manual tuning is limited:**
- You may overlook better configurations
- The search space grows quickly

**Search methods:**
- Random Search (used here)
- Bayesian Optimization (more sample-efficient)
- Hyperband (resource-efficient)

**Parameters tuned here:**
- Number of units in the hidden layer
- Learning rate of Adam

**How it works conceptually:**
1. Sample a hyperparameter configuration
2. Train the model for a few epochs
3. Evaluate validation loss
4. Keep the best configuration

## 10. Evaluation
Best hyperparameters are reported and then used to train a final model.

## 11. Results and Insights
- Tuning provides a data-driven choice of model size and learning rate
- Results can vary due to random initialization

## 12. Alternative Approaches
- Grid search for small search spaces
- Bayesian optimization for efficiency
- Manual tuning for quick prototypes

## 13. How to Run
1. Install dependencies:
   - `pip install tensorflow keras-tuner`
2. Prepare `x_train` and `y_train` in the notebook or import from a dataset
3. Open [Keras_Tuner.ipynb](Keras_Tuner.ipynb)
4. Run cells to perform the search and train the best model
