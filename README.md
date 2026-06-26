## ⚡ Project 01 — Smart Grid Failure Prevention Model

**`Python` `Scikit-learn` `Random Forest` `K-Means` `Google Colab`**

### Overview
An end-to-end predictive maintenance proof of concept built on 10,000 industrial sensor readings from electrical grid equipment. The system predicts equipment failures before they occur and segments healthy assets into prioritized maintenance tiers, shifting maintenance strategy from **reactive** (fix after failure) to **proactive** (prevent before failure).

> 💡 **Why this project?** As an EEE student, I wanted to apply ML to a domain I genuinely understand transformer failures, thermal stress, and equipment degradation. This sits at the intersection of my engineering background and data science interest.

---

### 📊 Dataset
- **Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Repository)
- **Size:** 10,000 sensor readings | 14 raw features → 5 final modeling features
- **Target:** Binary — Machine Failure (1) / No Failure (0)
- **Class Balance:** 96.6% No Failure / 3.4% Failure → severe imbalance handled via recall optimization

---

### 🔧 Full Pipeline

```
1. Data Cleaning          → Removed ID columns, dropped failure sub-type labels (leakage prevention)
2. Encoding               → One-hot encoding for machine type (L/M/H) with dummy variable trap handled
3. EDA                    → Correlation heatmap, sensor distributions, failure rate by machine type
4. Feature Selection      → VIF analysis removed Air Temp (VIF: 87K) and RPM (VIF: 315)
5. Train-Test Split       → 80/20 stratified split preserving 3.4% failure rate in both sets
6. Random Forest          → GridSearchCV with scoring='recall', class_weight='balanced'
7. Cross Validation       → StratifiedKFold (k=5) for reliable, imbalance-aware evaluation
8. ROC-AUC & Threshold    → Threshold tuned from 0.50 → 0.358 for optimal recall
9. K-Means Clustering     → Elbow + Silhouette method, K=3 via domain override
10. Visualization         → ROC curve, confusion matrix, cluster scatter plots, dashboard
11. Business Insights     → Maintenance schedule, cost-benefit analysis, recommendations
```

---

### 📈 Results

#### Failure Prediction (Random Forest)
| Stage | Recall | Notes |
|-------|--------|-------|
| Baseline RF (no tuning) | 38.2% | Starting point |
| Tuned RF (GridSearchCV) | 50.0% | +11.8% improvement |
| Cross-validated (StratifiedKFold) | 57.81% | Reliable, consistent estimate |
| Optimal threshold (0.358) | **73.5%** | Best operational performance |

| Metric | Value |
|--------|-------|
| ROC-AUC (single split) | **90.22%** |
| ROC-AUC (cross-validated) | **92.58%** |
| Optimal Threshold | **0.358** |
| False Alarm Rate at Optimal Threshold | **8.6%** |

#### Asset Segmentation (K-Means, K=3)
| Maintenance Tier | Inspection Frequency | Primary Signal |
|-----------------|---------------------|----------------|
| 🟢 Low Risk | Every 6 months | Low temp + low wear |
| 🟡 Medium Risk | Every 3 months | High tool wear hours |
| 🔴 High Risk | Monthly | High process temperature |

- **9,661 healthy assets** segmented into 3 actionable maintenance tiers
- **Primary clustering driver:** Process Temperature
- **Secondary clustering driver:** Tool Wear [min]

---

### 💡 Key Insights

- **Why Recall over Accuracy?** Missing a transformer failure (False Negative) risks blackouts and crores in emergency repair costs. A false alarm costs one unnecessary maintenance visit — a fraction of that. Asymmetric costs justify recall optimization.
- **Threshold tuning impact:** Lowering threshold from 0.50 → 0.358 catches ~16 additional failures per cycle at only 8.6% false alarm rate
- **Top failure predictors:** Torque [Nm] (corr: 0.19) and Tool Wear [min] (corr: 0.11)
- **Estimated savings:** Rs. 3 Lakh to 15+ Crore per maintenance cycle*

> *Cost estimates are illustrative projections based on publicly reported transformer failure costs, not measured outcomes from a live deployment.

---

### 🔍 Analytical Decisions Worth Noting

| Decision | Justification |
|----------|--------------|
| Dropped Air Temp (VIF: 87,435) | Multicollinear with Process Temp (corr: 0.88) — kept Process Temp as it directly reflects internal machine state |
| Dropped RPM (VIF: 315) | Multicollinear with Torque (corr: -0.88) — kept Torque as stronger failure predictor (0.19 vs -0.044) |
| Reverted feature selection | Dropping Type_L & Type_M reduced recall 50% → 44.3% — data-driven decision to revert |
| K=3 domain override | Silhouette suggested K=6 (score: 0.2563) but all K scores were narrow (0.23-0.26) — K=3 is operationally actionable |
| `scoring='recall'` in GridSearchCV | Directly optimizes for catching failures, not overall accuracy |

---

### 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Modeling, evaluation, clustering |
| Matplotlib & Seaborn | Visualization |
| Statsmodels | VIF calculation |
| Google Colab | Development environment |

---

### Limitations & Future Scope

**Current limitations:**
- Dataset is a benchmark dataset — real-world SCADA sensor data would produce stronger cluster separation and higher recall
- No temporal/time-series modeling — seasonal load patterns and degradation trends not captured
- Static threshold — would need periodic re-evaluation in production

**Future improvements:**
- LSTM or Prophet for time-series failure prediction
- SHAP values for individual machine failure explanation
- Real-time monitoring dashboard (Streamlit)
- XGBoost / LightGBM comparison
- Anomaly detection layer (Isolation Forest)

---

## 🧰 Common Tech Stack

```
Languages:    Python
ML / Stats:   Scikit-learn, Statsmodels, NumPy, Pandas
Visualization: Matplotlib, Seaborn
NLP:          TF-IDF, N-grams, Google Translate API
Environment:  Google Colab, Jupyter Notebook
```

---

---

*⭐ If you find these projects useful or interesting, a star on the repo would mean a lot!*
