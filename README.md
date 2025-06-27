---

## 📊 Dataset Summary

- **Rows**: 95,662  
- **Features**: 16  
- **No missing values**  
- Highly skewed `Amount` and `Value`  
- `FraudResult` is heavily imbalanced (proxy only)  
- `CountryCode` has no variance (removed)

---

## 🔧 Feature Engineering (Task 3)

All preprocessing is implemented in Python scripts within `src/` using `sklearn.pipeline.Pipeline`. Includes:

- **Aggregations**: total, mean, std dev transaction amounts per customer
- **Temporal**: transaction hour, day, month, year
- **Encoding**: label encoding (with plan for one-hot)
- **Scaling**: standardization & log transforms
- **Outlier detection**: handled using boxplots and IQR rules
- **IV & WOE**: Feature selection pipelines using `xverse`, `woe`

---

## 🧪 Proxy Target Engineering (Task 4)

- **RFM metrics**: Recency, Frequency, Monetary value per customer
- **Clustering**: KMeans clustering with scaled RFM features
- **Labeling**: Customers in least active cluster → `is_high_risk = 1`
- **Merged** back to main training set for model development

---

## 🤖 Model Training (Task 5)

- Models to be trained:
  - Logistic Regression
  - Random Forest
- Evaluation metrics:
  - Accuracy, Precision, Recall, F1, ROC-AUC
- Tracked using **MLflow**
- Hyperparameter tuning with **GridSearchCV**

---

## 🛠️ Deployment (Task 6)

- **API Framework**: FastAPI
- `/predict` endpoint returns risk probability
- **Model loading** from MLflow Registry
- **Validated input/output** via Pydantic
- Dockerized with `Dockerfile` and `docker-compose.yml`

---

## 🔄 CI/CD Pipeline

**CI on push to main:**

- Linting via `flake8`
- Unit testing via `pytest`

Configured in `.github/workflows/ci.yml`

---

## 📝 Interim Submission

- EDA completed
- Feature engineering in progress
- GitHub Actions partially configured
- Proxy risk design planned
- See full [Interim Report PDF](link-if-available)

---

## 📌 References

- [xverse](https://pypi.org/project/xverse/)
- [WOE & IV – Listendata](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [Investopedia: Credit Risk](https://www.investopedia.com/terms/c/creditrisk.asp)
- [Nova Challenge Guidelines](https://www.novaed.com)

---

## 👤 Author

**Miskir B.**  
KAIM – Week 5  
[GitHub Profile](https://github.com/MiskirB)

---
