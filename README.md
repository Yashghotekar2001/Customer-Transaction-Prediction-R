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

## Files in this repository

- `Customer-transaction-prediction-real.ipynb`  
  Jupyter notebook containing the full pipeline: EDA, preprocessing, model training, evaluation, feature importance, and saving artifacts.[1]
- `rfmodel.pkl`  
  Trained Random Forest model saved with `joblib.dump`.[1]
- `scaler.pkl`  
  Fitted `StandardScaler` used to transform features.[1]
- `featurecols.pkl`  
  List of feature column names used during training.[1]

## How to run

1. Clone the repository and open the notebook:

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   jupyter notebook Customer-transaction-prediction-real.ipynb
   ```

2. Install required Python packages (example):

   ```bash
   pip install numpy pandas scikit-learn seaborn matplotlib joblib
   ```

3. Run the notebook cells in order:
   - Load dataset (`train1.csv`).  
   - Perform EDA and preprocessing.  
   - Train the Random Forest model.  
   - Evaluate using AUC and classification report.  
   - Save model, scaler, and feature list.[1]

## Next improvements

- Try additional algorithms (e.g. XGBoost, LightGBM, Logistic Regression) and compare AUC and recall for class 1.[1]
- Perform hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.[1]
- Optimize decision threshold to balance precision and recall for class 1.[1]
- Add cross‑validation and a proper model comparison table in the notebook.[1]

***

If you want, the text can be shortened or translated to more informal language before you paste it into GitHub.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153745845/97a07cfa-3262-4b5f-8992-35e99c97e038/Customer-transaction-prediction-real.ipynb)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153745845/b937082e-1c3a-4275-ba2e-c52b1afceb3b/insurance-cost-prediction-real.ipynb)# Customer-Transaction-Prediction-R
This project predicts whether a customer will make a future transaction using a highly imbalanced, anonymized financial dataset with 200 numerical features and a binary target. The data contains 200,000 rows, no missing values, and a target distribution of roughly 90% class 0 (no transaction) and 10% class 1 (transaction), which makes handling
