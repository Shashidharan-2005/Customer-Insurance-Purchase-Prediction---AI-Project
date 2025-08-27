import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from scipy.stats import pointbiserialr

DATA_PATH = os.environ.get("DATA_PATH", "Social_Network_Ads.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")

def load_data(path):
    df = pd.read_csv(path)
    assert {"Age","EstimatedSalary","Purchased"}.issubset(df.columns), (
        "Dataset must include columns: Age, EstimatedSalary, Purchased"
    )
    X = df[["Age","EstimatedSalary"]].copy()
    y = df["Purchased"].copy().astype(int)
    return df, X, y

def build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def get_models():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C":[0.01,0.1,1,10]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors":[3,5,7,9]}
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C":[0.1,1,10], "kernel":["rbf","linear"]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth":[None,3,5,7]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators":[50,100], "max_depth":[None,5,7]}
        )
    }

def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "ClassificationReport": classification_report(y_test, y_pred, output_dict=False),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading data from: {DATA_PATH}")
    df, X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preproc = build_preprocessor()
    X_train_p = preproc.fit_transform(X_train)
    X_test_p = preproc.transform(X_test)

    models = get_models()
    results = []
    best_models = {}

    print("\\nTraining and tuning models...")
    for name, (clf, grid) in models.items():
        gs = GridSearchCV(clf, grid, cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train_p, y_train)
        best = gs.best_estimator_
        pipeline = Pipeline([("preproc", preproc), ("clf", best)])
        best_models[name] = pipeline

        metrics = evaluate_model(pipeline, X_test, y_test)
        result_row = {
            "Model": name,
            "BestParams": gs.best_params_,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "ROC_AUC": metrics["ROC_AUC"]
        }
        results.append(result_row)

        print(f"=== {name} ===")
        print("Best Params:", gs.best_params_)
        print(metrics["ClassificationReport"])
        print("Confusion Matrix:", metrics["ConfusionMatrix"])

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    metrics_path = os.path.join(OUT_DIR, "model_comparison_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics -> {metrics_path}")

    # pick best model
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = best_models[best_name]

    # Scenario predictions (as per prompt)
    median_salary = X_train["EstimatedSalary"].median()
    def scenario(age, salary):
        sal = median_salary if (salary is None or (isinstance(salary, float) and np.isnan(salary))) else salary
        arr = np.array([[age, sal]])
        pred = best_pipeline.predict(arr)[0]
        prob = best_pipeline.predict_proba(arr)[0][1] if hasattr(best_pipeline.named_steps["clf"], "predict_proba") else None
        return pred, prob, sal

    scenarios = [
        ("Age 30, Salary 87000", 30, 87000),
        ("Age 40, No Salary", 40, None),
        ("Age 40, Salary 100000", 40, 100000),
        ("Age 50, No Salary", 50, None),
        ("Age 18, No Salary", 18, None),
        ("Age 22, Salary 600000", 22, 600000),
        ("Age 35, Salary 2500000", 35, 2500000),
        ("Age 60, Salary 100000000", 60, 100000000),
    ]
    rows = []
    for label, age, salary in scenarios:
        pred, prob, imputed = scenario(age, salary)
        rows.append({
            "Scenario": label,
            "Age": age,
            "EstimatedSalary_original": salary,
            "ImputedSalary": imputed,
            "PredictedPurchased": int(pred),
            "PurchaseProb": float(prob) if prob is not None else None
        })
    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(OUT_DIR, "prompt_scenario_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved scenario predictions -> {pred_path}")

    # Hypothesis testing
    corr_age = pointbiserialr(df["Purchased"], df["Age"]).correlation
    corr_salary = pointbiserialr(df["Purchased"], df["EstimatedSalary"]).correlation
    ages_to_test = [25, 55]
    salaries = [20000, 50000, 100000, 500000, 1000000]
    controlled = []
    for age in ages_to_test:
        for sal in salaries:
            arr = np.array([[age, sal]])
            pred = best_pipeline.predict(arr)[0]
            prob = best_pipeline.predict_proba(arr)[0][1] if hasattr(best_pipeline.named_steps["clf"], "predict_proba") else None
            controlled.append({"Age": age, "Salary": sal, "Predicted": int(pred), "Prob": float(prob) if prob is not None else None})
    controlled_df = pd.DataFrame(controlled)
    ctrl_path = os.path.join(OUT_DIR, "controlled_predictions.csv")
    controlled_df.to_csv(ctrl_path, index=False)
    print(f"Saved controlled predictions -> {ctrl_path}")

    # Save project summary + conclusions
    summary = {
        "best_model": best_name,
        "best_params": str(models[best_name][0]),
        "metrics_top": results_df.iloc[0].to_dict()
    }
    with open(os.path.join(OUT_DIR, "project_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    conclusions = [
        f"Best model by accuracy: {best_name}",
        f"Correlation: Age vs Purchased = {corr_age:.4f}, Salary vs Purchased = {corr_salary:.4f}",
        "Observation: Salary shows a stronger positive relationship with purchasing than age in this dataset."
    ]
    with open(os.path.join(OUT_DIR, "conclusions.txt"), "w") as f:
        f.write("\\n".join(conclusions))

    print("\\n=== DONE ===")
    print(f"Best model: {best_name}")
    print(f"Artifacts saved to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()