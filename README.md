# Multiple Linear Regression Analysix

This repository contains a Jupyter Notebook (**multiple_linear_regression.ipynb**) that demonstrates the implementation and evaluation of a **Multiple Linear Regression** model using Python and the **scikit-learn** library.

The notebook uses **synthetic data** generated specifically for this demonstration, featuring **two predictor variables (features)** and **one target variable**.

---

## 🚀 Project Goal

The primary goal of this notebook is to:

- Generate a **synthetic dataset** suitable for multiple linear regression.  
- **Train** a Linear Regression model on the dataset.  
- **Evaluate** the model's performance using standard regression metrics.  
- **Visualize** the data and, potentially, the regression plane (though not explicitly shown in output, it is a common step with 2 features).

---

## 🛠️ Dependencies and Requirements

This notebook requires the following Python libraries. You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn plotly
```
 - **Library	Purpose**
- **pandas**
- Data structuring and manipulation (e.g., creating a DataFrame with feature1, feature2, and target columns).
- **numpy**
- Numerical operations and array manipulation.
- **scikit-learn**
- Regression model (LinearRegression), data splitting (train_test_split), and metric calculation (mean_absolute_error, mean_squared_error, r2_score).
- **plotly**
- Interactive visualization (plotly.express, plotly.graph_objects).

# 📘 Dataset Details
- The dataset is synthetically generated using sklearn.datasets.make_regression with the following parameters:
- Number of Samples (n_samples): 250
- Number of Features (n_features): 2
- Noise (noise): 50 (indicating the degree of scatter around the true linear relationship).
- Data Structure: A Pandas DataFrame with columns — feature1, feature2, and target.

# 📊 Key Steps and Results
- **1. Data Preparation**
- A dataset with 250 samples and 3 columns is created and inspected.
- Features are labeled feature1 and feature2, and the dependent variable is target.

**2. Model Training**
- A LinearRegression model is initialized and trained on the dataset using:
```bash
model = LinearRegression()
model.fit(X_train, y_train)
```
- The model learns coefficients that minimize the residual sum of squares between the observed and predicted values.
  
**3. Coefficient Inspection**
- After training, the coefficients ($\beta_i$) learned by the model can be inspected to understand how each feature contributes to the target variable.
```bash
Example Output (Coefficients):
array([59.2562287 , 5.21846039])
```
- These values show that feature1 has a higher weight (stronger influence) than feature2.

**4. Evaluation**
- The model’s performance is evaluated using standard regression metrics:

**Metric	Description**
- Mean Absolute Error (MAE)	Average magnitude of prediction errors in the same units as the target variable.
- Mean Squared Error (MSE)	Average of squared prediction errors, penalizing large deviations.
- R² Score (R2)	Measures how much variance in the target variable is explained by the features.

**🧮 Mathematical Representation**
For a Multiple Linear Regression model:

**𝑦 = 𝛽0 + 𝛽1𝑥1 + 𝛽2𝑥2 + 𝜖**

Where:
- 𝑦 → target variable
- 𝑥1, 𝑥2 → predictor variables
- 𝛽0 → intercept
- 𝛽1, 𝛽2 → feature coefficients (weights)
- ϵ → residual error term

- The model finds the best-fit hyperplane that minimizes the sum of squared residuals.

# 💡 Insights and Takeaways
- Multiple Linear Regression helps understand how multiple factors simultaneously influence a target variable.
- Inspecting coefficients reveals relative importance of predictors.
- Adding too many features without sufficient data can lead to overfitting — use Adjusted R² to evaluate fairly.
- Noise controls realism — higher noise simulates real-world imperfections in data.
- Visualizing in 3D (feature1, feature2, target) using Plotly can make regression results more interpretable.

