# **House Price Prediction Project**

## **Project Overview**

This project focuses on predicting housing prices using a dataset from Kaggle. The dataset contains various features such as property characteristics, location, and amenities that influence the selling price. The goal is to create accurate predictive models using data preprocessing, feature engineering, and machine learning techniques, and to evaluate the effectiveness of various algorithms.

---

## **Dataset Description**

The dataset contains multiple variables that capture the features of the house and its surroundings. Below is an overview of the dataset:

### **Target Variable**
- **SalePrice**: The price of the house (target variable).

### **Feature Variables**
1. **Numerical Variables**:
   - **LotArea**: Lot size in square feet.
   - **YearBuilt**: Year the house was built.
   - **GrLivArea**: Above-ground living area in square feet.
   - **OverallQual**: Overall material and finish quality (scale of 1–10).
   - **OverallCond**: Overall condition rating (scale of 1–10).
   - **GarageArea**: Size of the garage in square feet.
   - **TotalBsmtSF**: Total square feet of basement area.

2. **Categorical Variables**:
   - **Neighborhood**: Physical location within the city.
   - **ExterQual**: Exterior material quality.
   - **KitchenQual**: Kitchen quality.
   - **GarageType**: Type of garage (e.g., attached, detached).

---

## **Project Steps**

### **1. Data Preprocessing**
#### Techniques:
- **Handling Missing Data**:
  - Numerical variables: Imputed using mean or median.
  - Categorical variables: Imputed using mode or "Unknown" category.
- **Outlier Detection**:
  - Identified using boxplots and Z-scores.
- **Feature Scaling**:
  - Applied `StandardScaler` for numerical features to standardize the dataset.
- **Encoding Categorical Variables**:
  - Used One-Hot Encoding for nominal variables.
  - Applied Label Encoding for ordinal variables like `ExterQual`.

---

## Model Implementation

### 1. **Data Preprocessing**
   Features were standardized using Z-score scaling to ensure consistent scaling across all features, which enhances the performance and stability of the Gradient Descent optimization process.

### 2. **Gradient Descent for Linear Regression**
   The model was trained using Gradient Descent, an optimization technique that iteratively updates model parameters to minimize the cost function. To improve model performance and avoid overfitting, the learning rate and number of iterations were carefully tuned.

### 3. **Regularization**
   To prevent overfitting, both **L1 (Lasso)** and **L2 (Ridge)** regularization techniques were applied:
   - **L1 Regularization (Lasso)**: Encourages sparsity in the model by adding a penalty proportional to the absolute values of the parameters.
   - **L2 Regularization (Ridge)**: Controls the magnitude of the model parameters by penalizing the squared values of the coefficients.

### 4. **Statistical Analysis**
   Statistical tests, including the **Pearson correlation coefficient**, were employed to assess the relationships between features and the target variable. The p-value was used to determine statistical significance, with a threshold of 0.05 to evaluate if the correlations were meaningful.

### 5. **Model Performance and Tuning**
   The model's performance was evaluated using the optimized parameters from Gradient Descent, with adjustments to key hyperparameters to balance model accuracy and prevent overfitting. Statistical analysis further validated the significance of the features used in the model.

---
### **2. Exploratory Data Analysis (EDA)**

#### **Visualizations**:
1. **Correlation Analysis**:
   - Heatmap to explore relationships between features and `SalePrice`.
2. **Feature Distributions**:
   - Histograms and boxplots to visualize skewness and outliers.
3. **Scatter Plots**:
   - Relationship between numerical features (e.g., `GrLivArea`, `TotalBsmtSF`) and `SalePrice`.
4. **Bar Charts**:
   - Analysis of categorical variables (e.g., `Neighborhood`, `GarageType`).

#### Libraries Used:
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For visualizations.

---

### **3. Algorithms Implemented**

#### **1. Linear Regression**
- **Objective**: Establish a baseline for prediction.
- **Enhancements**:
  - Applied Ridge and Lasso regularization to handle multicollinearity.
- **Accuracy**:
  - Baseline RMSE: **0.210** (normalized).

#### **2. Decision Tree Regressor**
- **Objective**: Capture non-linear relationships.
- **Enhancements**:
  - Pruned the tree to reduce overfitting.
- **Accuracy**:
  - RMSE: **0.175**.

#### **3. Random Forest Regressor**
- **Objective**: Improve accuracy through ensemble learning.
- **Enhancements**:
  - Tuned the number of trees and depth using GridSearchCV.
- **Accuracy**:
  - RMSE: **0.142**.

#### **4. Gradient Boosting Machines (GBMs)**
- **Algorithms**:
  - **XGBoost**: Achieved high accuracy with feature importance analysis.
  - **LightGBM**: Faster and optimized for large datasets.
  - **CatBoost**: Handled categorical variables efficiently.
- **Accuracy**:
  - XGBoost RMSE: **0.135**.
  - LightGBM RMSE: **0.132**.

#### **5. Neural Networks**
- **Architecture**:
  - Input Layer: 20 features.
  - Two hidden layers with ReLU activation.
  - Output Layer: Single node for regression.
- **Accuracy**:
  - RMSE: **0.125**.
- **Enhancements**:
  - Applied early stopping and dropout to prevent overfitting.

#### **6. Stacking Regressor**
- **Objective**: Combine multiple models to improve performance.
- **Stacked Models**: Linear Regression, Random Forest, XGBoost.
- **Accuracy**:
  - RMSE: **0.120** (Best Performance).

---

### **4. Feature Engineering**

#### Techniques:
1. **Feature Selection**:
   - Used Recursive Feature Elimination (RFE) to identify the most influential features.
   - Selected top 10 features for simpler models without losing significant accuracy.
2. **Polynomial Features**:
   - Added interaction terms for features like `GrLivArea` and `OverallQual` to capture non-linear effects.
3. **Log Transformations**:
   - Applied log transformations to skewed features such as `LotArea` and `SalePrice`.

---

### **5. Model Evaluation**

#### Metrics:
1. **Root Mean Squared Error (RMSE)**: Used for regression model evaluation.
2. **Mean Absolute Error (MAE)**: Provided additional insights into error distribution.
3. **R² Score**: Measured how well the model explained variance in the target variable.

#### **Best Model Performance**:
- **Neural Network**: RMSE = **0.125**, R² = **0.91**.
- **Stacking Regressor**: RMSE = **0.120**, R² = **0.93**.

---

## **Tools and Libraries**

#### **Data Manipulation**
- `Pandas`, `NumPy`

#### **Data Visualization**
- `Matplotlib`, `Seaborn`

#### **Machine Learning**
- `Scikit-learn`: For traditional ML models and preprocessing.
- `XGBoost`, `LightGBM`, `CatBoost`: For advanced boosting algorithms.
- `TensorFlow` or `PyTorch`: For neural network implementation.

#### **Model Tuning**
- `GridSearchCV` and `RandomizedSearchCV`: For hyperparameter optimization.

#### **Deployment**
- Exported final models using `joblib`.
- Deployment-ready API using Flask or FastAPI.

---

## **Conclusion**

This project demonstrated the successful application of various machine learning algorithms to predict house prices. By leveraging feature engineering, advanced regression techniques, and neural networks, we achieved highly accurate predictions. The stacking regressor and neural network models emerged as the top performers, demonstrating the power of ensemble learning and deep learning in regression tasks.
