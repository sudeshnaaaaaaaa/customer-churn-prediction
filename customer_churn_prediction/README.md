# Customer Churn Prediction

An end-to-end machine learning project: cleans and explores customer data, engineers features, trains and compares three classification models, and evaluates them with metrics suited to an imbalanced target.

## A note on the data — read this first

The dataset (`customer_churn.csv`) is **synthetically generated** (see `generate_data.py`), not pulled from a public source like Kaggle. It's built with realistic, correlated churn drivers (contract type, tenure, support calls, etc.) so the model has genuine signal to learn from — but be upfront about this if asked. It's honest and still demonstrates the full skillset; it's just not a live business dataset.

**If you want to swap in a real dataset later:** the IBM Telco Customer Churn dataset on Kaggle has an almost identical schema — download it and point `churn_prediction.ipynb` at it with minimal changes.

## What's in this project

- `generate_data.py` — builds the synthetic dataset with realistic churn patterns
- `customer_churn.csv` — the generated dataset (2,000 customers, ~27% churn rate)
- `churn_prediction.ipynb` — the full pipeline: EDA → feature engineering → model training → evaluation → feature importance

## Pipeline summary

1. **Cleaning:** imputed a small number of missing `total_charges` values from `tenure × monthly_charges`.
2. **EDA:** found month-to-month contracts, short tenure, and frequent support calls all associate with higher churn — visualized with bar charts and box plots.
3. **Feature engineering:** one-hot encoded categorical fields, added an `avg_monthly_spend` ratio feature.
4. **Modeling:** trained Logistic Regression, Random Forest, and Gradient Boosting; used `class_weight="balanced"` to handle the imbalanced target rather than relying on raw accuracy.
5. **Evaluation:** compared F1 (churn class) and ROC-AUC across all three models rather than just accuracy, since a model that always predicts "no churn" would already look ~73% accurate while being useless.
6. **Result:** Logistic Regression performed best on this run (ROC-AUC ≈ 0.69, F1 for the churn class ≈ 0.50) — re-run the notebook to reproduce or vary these numbers.

## Interview talking points

- **Why not just use accuracy?** With ~27% churn, a trivial "predict no-churn always" model hits ~73% accuracy while catching zero actual churners. F1 and ROC-AUC on the churn class are what matter here.
- **Why `class_weight="balanced"`?** It penalizes misclassifying the minority (churn) class more heavily during training, instead of letting the model default to the majority class.
- **Why compare three models instead of picking one upfront?** Different models make different bias/variance trade-offs — logistic regression is interpretable and worked best here, but that isn't guaranteed on every dataset, so comparing is the right default habit.
- **What would you change for production?** Validate on a genuinely time-based holdout (not a random split, since customer behavior drifts over time), and tune the classification threshold based on the real business cost of a false negative (a lost customer) vs. a false positive (an unnecessary retention offer).

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib
python generate_data.py        # regenerates customer_churn.csv
jupyter notebook churn_prediction.ipynb
```
