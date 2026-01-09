***

# Customer Transaction Prediction

This project builds a machine learning model to predict whether a customer will make a future transaction based on 200 anonymized numerical features.[1]

## Project overview

- Problem type: Binary classification (target 0 = no transaction, 1 = transaction).[1]
- Data size: 200,000 rows and 200 feature columns (after dropping `ID_code`).[1]
- Goal: Identify high‑risk / likely‑to‑transact customers so the business can focus marketing or risk strategies on them.[1]

## Data description

- Original columns:
  - `ID_code`: unique row identifier (dropped, not useful for prediction).[1]
  - `target`: 0 or 1 indicating whether a customer made a transaction.[1]
  - `var_0` to `var_199`: anonymized numerical features.[1]
- Class balance:
  - Around 90% of records are class 0, 10% are class 1 (highly imbalanced).[1]
- Data quality:
  - No missing values in any column.[1]

## Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Checked shape, dtypes, and basic statistics for selected variables.[1]
   - Verified no null or missing values.[1]
   - Plotted distributions of features (e.g., `var0`, `var1`) split by `target` to understand separation between classes.[1]

2. **Preprocessing**  
   - Dropped `ID_code` as it is only an identifier.[1]
   - Separated features and label:
     - `X`: all `var_*` columns.  
     - `y`: `target` converted to integer.[1]
   - Train–test split:
     - 80% train, 20% test.  
     - Used `stratify=y` to keep the same 0/1 ratio in train and test.[1]
   - Feature scaling:
     - Applied `StandardScaler` on training data and used the same scaler for test data.[1]

3. **Modeling – Random Forest Classifier**  
   - Algorithm: `RandomForestClassifier`.[1]
   - Key hyperparameters:
     - `n_estimators=300`  
     - `random_state=42`  
     - `n_jobs=-1` (use all CPU cores)  
     - `class_weight='balanced'` to give more weight to minority class 1.[1]
   - Trained the model on scaled training data.[1]

4. **Model evaluation**  
   - Predicted probabilities for class 1 on the test set.[1]
   - Metrics:
     - AUC (ROC): about **0.82** on the test set.[1]
     - ROC curve plotted to visualize trade‑off between true positive rate and false positive rate.[1]
   - Classification report (threshold 0.5):
     - High accuracy due to class imbalance but very low recall for class 1 at 0.5 threshold.[1]
     - Notes in notebook suggest trying lower thresholds (0.4, 0.3) to improve recall.[1]

5. **Feature importance**  
   - Extracted feature importances from the trained Random Forest.[1]
   - Plotted top 20 most important `var_*` features (e.g., `var81`, `var139`, `var12`, `var110`, `var53`).[1]
   - Printed the top 10 important features with their importance scores.[1]
***
