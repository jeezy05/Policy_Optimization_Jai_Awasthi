# Loan Default Prediction and RL Policy Optimization

A comprehensive machine learning project for loan default prediction and policy optimization using Deep Learning and Offline Reinforcement Learning on Lending Club data.

---

## Project Overview

This repository demonstrates:

- **Exploratory Data Analysis (EDA):**  
  Detailed analysis of loan data—missing values, feature distributions, categorical effects, correlation, and data leakage mitigation.
- **Deep Learning Model:**  
  Tabular MLP classifier in PyTorch for binary loan default prediction.
- **Offline RL Policy:**  
  DQN agent learns optimal loan approval decisions using historic data and customized reward functions.

---

## Data & Structure

- **Source:** Lending Club (2007-2018 Q4)
- **Size:** Approximately 559,000 records with 150+ features
- **Key Files:**
  cleaned/df_clean_intermediate.csv # Cleaned features
  cleaned/train_clean.csv # Training split
  cleaned/test_clean.csv # Testing split
  data/train_mdp.pkl # RL train dataset
  data/test_mdp.pkl # RL test dataset
  data/best_mlp.pt # DL model checkpoint
  notebook.ipynb # Main notebook

---

## Requirements

pip install pandas numpy matplotlib seaborn scikit-learn torch d3rlpy scipy statsmodels

- Python ≥ 3.8

---

## 1. Exploratory Data Analysis (EDA)

- Identified and dropped columns with over 80% missing values
- Engineered features: loan-to-income ratio, log transformations, credit age
- Risk Drivers: grade, interest rate, and term highly associated with default risk
- Detected and removed highly correlated columns to avoid data leakage
- Class imbalance: defaults approximately 20%

### Output

- Missing value heatmaps
- Distribution and boxplots
- Correlation matrix
- EDA summary in `EDA_short_report.csv`

---

## 2. Deep Learning Classification

- **Model:**  
  Multi-layer MLP with BatchNorm, ReLU, Dropout
- **Imbalance Handling:**  
  Weighted BCE loss
- **Results:**  
  | Metric    | Value  |
  |-----------|--------|
  | ROC AUC   | 0.71   |
  | Precision | 0.28   |
  | Recall    | 0.69   |
  | F1-score  | 0.40   |
  | Accuracy  | 0.63   |

- **Interpretation:**  
  The model achieves high recall for defaults, suitable for conservative risk management.

---

## 3. Offline RL Policy (DQN)

- **State:** Applicant and loan features
- **Action:** Approve (1) or Deny (0)
- **Reward:**  
  - Approved and paid: positive reward (interest earned)
  - Approved and defaulted: negative reward (loss on principal)
  - Denied: zero reward
- **Training:**  
  DQN (d3rlpy), gamma=1.0, batch size=256, 100,000 training steps

### Results & Policy Behavior

| Policy             | Approval % | Mean Reward | Status     |
|--------------------|------------|-------------|------------|
| Always Approve     |   100      | –0.0057     | Loss-making|
| DQN RL Policy      |   52       | +0.0214     | Profitable |

- RL policy is conservative and generally more profitable than naïve approval strategies.

---

## Reproducibility

1. Run each notebook section in sequence after placing the dataset in the repository root.
2. Outputs are saved in `/cleaned` and `/data` directories at each stage.

---

## Key Insights

- Drop sparse and leakage features early in the workflow
- Deep learning baseline achieves solid recall for risky loans
- Offline RL agent learns profitable approval strategies that balance risk and reward

---

## Future Directions

- Explore alternative offline RL algorithms (e.g., CQL, BCQ)
- Add interpretability analysis (SHAP, LIME, feature importance)
- Refine reward engineering for different business objectives
- Evaluate and address fairness considerations

---

## License

For educational and research use.

**Author:** Jai Awasthi

