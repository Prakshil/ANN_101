# Feature Scaling in Artificial Neural Networks

## Overview
This folder contains the `feature_scaling.ipynb` notebook which practically demonstrates why data normalization is crucial when training Neural Networks. It highlights the impact of features having vastly different scales on model convergence and overall accuracy.

## Project Structure
- `feature_scaling.ipynb`: The main notebook demonstrating the effects of scaling.
- `Social_Network_Ads.csv`: The dataset used, where features like Age and Estimated Salary are on entirely different magnitudes.

## Key Concepts Demonstrated
1. **Unscaled Trouble**: Training on raw features often leads to slow convergence and poor validation curves due to the optimizer struggling in an asymmetrical, elongated cost function bowl.
2. **Standardization (`StandardScaler`)**: Adjusting features to have a mean of 0 and standard deviation of 1.
3. **Scaled Performance**: Shows how identical network architectures learn much faster and reach higher accuracy cleanly when features are normalized to roughly the same scale.

## Instructions
1. Install prerequisites: `pip install pandas numpy scikit-learn tensorflow seaborn matplotlib`
2. Open `feature_scaling.ipynb`.
3. Follow the step-by-step markdown to see training curves comparing Unscaled vs Scaled inputs.