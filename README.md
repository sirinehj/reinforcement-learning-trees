# Reinforcement Learning Trees (RLT)

**Non-Greedy Decision Trees Guided by Reinforcement Learning**

## Project Overview

Reinforcement Learning Trees (RLT) is an advanced tree-based learning framework designed for **classification** and **regression** tasks. Unlike classical greedy decision trees, RLT approaches tree construction as a **sequential decision-making problem**, optimizing split decisions based on **long-term predictive rewards** rather than local impurity reduction. 

This project adheres to the CRISP-DM methodology, ensuring a structured and rigorous process from problem formulation through deployment and evaluation.

---

## Motivation

Traditional tree-based models (e.g., CART, Random Forests, Gradient Boosting) rely on greedy, local split criteria, which may be insufficient in scenarios such as:
- **Weak marginal feature effects**
- **Strong feature correlations (multicollinearity)**
- **Non-linear interactions**
- **High-dimensional inputs**

Reinforcement Learning Trees address these limitations through:
- Look-ahead optimization
- Feature muting
- Embedded local models

This enables RLT to provide **robust learning in complex environments**.

---

## CRISP-DM Workflow

### 3.1 Business Understanding

The objectives of the project include:
- Improving robustness over traditional greedy tree-based methods
- Capturing interaction-driven signals
- Maintaining competitive performance in simple linear scenarios

---

### 3.2 Data Understanding

10 real-world datasets were used including:
1. **Sonar Dataset**: A classification task 
2. **Concrete Compressive Strength Dataset**: A regression task

Exploratory analysis highlighted the following:
- High dimensionality
- Correlated predictors
- Non-linear relationships

---

### 3.3 Data Preparation

Steps taken for data preparation:
- Consistent **train–test splits**
- **Feature scaling** for numerical variables
- Conservative outlier handling
- Avoidance of excessive feature elimination to allow intrinsic feature selection

---

### 3.4 Modeling

The model leverages an **ensemble of Reinforcement Learning Trees**, with the following unique design elements:
- **Node-level reinforcement learning**:
  - **State**: Data subset + historical tree information
  - **Actions**: Split choice, feature weighting, muting, protection
  - **Reward**: Downstream predictive performance
- Embedded **local models** (e.g., linear/regularized models) at the nodes

---

### 3.5 Evaluation

Evaluation metrics:
- **Classification**: Accuracy
- **Regression**: R² score

The RLT framework is benchmarked against traditional baselines:
- Ridge Regression (Regression) / Logistic Regression (Classification)
- Decision Trees
- Random Forests
- Extra Trees
- Gradient Boosting
- XGBoost
- Neural Networks (MLP)

---

### 3.6 Deployment

Deployment features include:
- **Reproducible experimentation** with MLflow for tracking
- **Model serving** with FastAPI
- Task automation with a **Makefile**
- Optional React-based user interface

---

## Evaluation Scenarios

The project includes both **synthetic** and **real-world** scenarios to evaluate:
- Linear benchmarks (Lasso-friendly settings)
- Non-linear regression
- Threshold effects
- Interaction structures with correlated features (e.g., checkerboard-type)

Key aspects tested:
- **Bias–variance tradeoff**
- **Interaction detection**
- **Robustness to noise and multicollinearity**

---

## Results Summary

Reinforcement Learning Trees demonstrate:
- Competitive or superior performance compared to traditional greedy tree-based models in complex scenarios
- **Robustness to weak marginal effects** and correlated predictors
- Effective capture of **non-linear relationships** and interaction-driven signals
- Competitive performance in simple **linear benchmark scenarios**

---

## Project Structure

- `src/`: Contains the implementation of RLT and supporting modules.
- `notebooks/`: Jupyter Notebooks for exploratory analysis and evaluation.
- `data/`: Preprocessed datasets used in the project.
- `results/`: Evaluation results and logs.
- `deployment/`: FastAPI code for serving the model (optional React interface).
- `Makefile`: Automation tasks (e.g., training, evaluation, deployment).

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/reinforcement-learning-trees.git
   cd reinforcement-learning-trees
   ```

2. Set up the Python environment:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Train and evaluate the model:
   ```bash
   make train
   make evaluate
   ```

4. Serve the model:
   ```bash
   make deploy
   ```

---

## Contributions

Contributions, improvements, and feature suggestions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
