# Employee Turnover Analytics

Machine Learning course-end capstone project for Portobello Tech's HR Department, following the course's [problem statement](1739525570_employee_turnover_problem_statement.pdf). Predicts which employees are at risk of leaving and turns that prediction into a risk-zone-based retention strategy.

## Problem

Portobello Tech's HR department wants to understand why valuable employees leave, and act before they do. This project analyzes 15,000 employee records to:

- Identify which factors most strongly drive turnover
- Cluster the employees who *already* left into distinct profiles (e.g. burnout vs. career growth)
- Train and compare classification models to predict who is at risk
- Convert model output into a practical, risk-zone-based retention strategy

## Data

[`HR_comma_sep.csv`](HR_comma_sep.csv) — 14,999 employee records, one row per employee:

| Column | Description |
|---|---|
| `satisfaction_level` | Employee-reported satisfaction (0-1) |
| `last_evaluation` | Score from most recent performance review (0-1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly working hours |
| `time_spend_company` | Tenure in years |
| `Work_accident` | Whether the employee had a workplace accident (0/1) |
| `left` | Target — whether the employee left (0/1) |
| `promotion_last_5years` | Whether promoted in the last 5 years (0/1) |
| `sales` | Department |
| `salary` | Salary tier (low/medium/high) |

## Approach

1. **EDA** — correlation heatmap, feature distributions, project count vs. attrition, and turnover rate broken down by department and salary tier.
2. **Clustering** — KMeans on employees who left, using satisfaction and evaluation score, with the cluster count (*k*=3) validated by silhouette score. Produces three interpretable profiles: burned-out high performers, satisfied high performers likely leaving for growth, and a mid-satisfaction/mid-evaluation group with no clear driver from these two features alone.
3. **Class imbalance** — categorical encoding, an 80:20 stratified train/test split, then SMOTE applied to the training set only (test set left untouched to avoid leakage).
4. **Model training** — 5-fold cross-validated Logistic Regression, Random Forest, and Gradient Boosting.
5. **Model selection** — compared ROC/AUC and confusion matrices on the held-out test set. Selected **recall** as the deciding metric (a missed leaver is far costlier than an unnecessary retention nudge), which favored **Gradient Boosting** (92.0% recall) over Random Forest (90.5% recall).
6. **Retention strategy** — scored the test set with predicted turnover probability, bucketed employees into Safe/Low/Medium/High risk zones, and mapped each zone (plus each cluster profile) to a specific recommended action.

## Key findings

- `satisfaction_level` and `time_spend_company` are the dominant predictors, together accounting for over 78% of feature importance — `number_project`, `salary`, and department matter, but are secondary levers.
- Salary shows a clean, monotonic relationship with turnover: 20.5% (low) → 14.6% (medium) → 4.8% (high).
- Project count has a U-shaped relationship with attrition — highest at 2 projects, near-zero at 3, climbing again from 6-7.
- Medium/High-risk employees benefit most from direct satisfaction check-ins and tenure-aware attention, with project-load rebalancing and salary review as supporting levers.

## Repo contents

| File | Description |
|---|---|
| [`Employee_Turnover_Analytics.ipynb`](Employee_Turnover_Analytics.ipynb) | Main analysis notebook (source of truth) |
| `HR_comma_sep.csv` | Raw dataset |
| `1739525570_employee_turnover_problem_statement.pdf` | Original course assignment brief |
| `Employee_Turnover_Technical_Report.pdf` / `.docx` | Written technical report |
| `Employee_Turnover_NonTechnical_Summary.pdf` / `.pptx` | Stakeholder-facing summary deck |
| `Submission/` | Final graded submission package (notebook, PDFs, screenshots) |

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook Employee_Turnover_Analytics.ipynb
```

Run all cells top to bottom; the notebook reads `HR_comma_sep.csv` from the same directory and produces all charts inline.
