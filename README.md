# Social Network Ads – ML Project

## How to run (quick)
```bash
# 1) Create venv (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the full pipeline
python run_project.py
```

Artifacts will be saved in the same folder:
- model_comparison_metrics.csv
- prompt_scenario_predictions.csv
- controlled_predictions.csv
- project_summary.json
- conclusions.txt

## Notebook
Open `Project.ipynb` to explore the analysis step-by-step with plots and tables.

## Notes
- Dataset expected: `Social_Network_Ads.csv` with columns `Age`, `EstimatedSalary`, `Purchased`.
- No personal identifiers (e.g., `User ID`) are used for modeling.