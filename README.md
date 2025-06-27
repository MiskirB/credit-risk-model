## 📊 Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord mandates that banks align their capital requirements with their risk exposure, particularly through Internal Ratings-Based (IRB) approaches. For this to be regulatory compliant, the credit risk models must be not only statistically sound but also **transparent, interpretable, and well-documented**.

In this context, interpretable models like **Logistic Regression with Weight of Evidence (WoE)** offer traceable logic that regulators and auditors can easily follow. Documentation and version control are critical for **model validation**, **internal auditing**, and **compliance reporting**. Black-box models, such as gradient boosting, may be rejected or require explainability techniques like SHAP or LIME to justify predictions in high-stakes environments like lending and credit scoring.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our dataset, there is no explicit "default" or "non-payment" outcome variable, which is essential for supervised learning. To resolve this, we create a **proxy target variable** using **RFM segmentation** and clustering. Customers who exhibit low transaction frequency, small monetary value, and high recency (inactivity) can be flagged as likely high-risk.

This proxy enables model training but comes with risks:

- **Mislabeling:** Customers may be classified high-risk due to behavior unrelated to creditworthiness.
- **Bias:** If the proxy is biased (e.g., penalizes seasonal users), the model may inherit that bias.
- **Business impact:** Poor targeting may lead to **denied loans for creditworthy customers** or **approved loans for high-risk customers**, both of which damage business performance and reputation.

Proper **validation, testing, and iteration** with business stakeholders is critical to mitigating these risks.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

There is a classic trade-off between **model interpretability** and **predictive power**:

| Aspect                | Logistic Regression (with WoE) | Gradient Boosting (e.g., XGBoost) |
| --------------------- | ------------------------------ | --------------------------------- |
| Interpretability      | ✅ Very high                   | ❌ Low (unless using SHAP)        |
| Performance           | ✅ Good (baseline)             | ✅✅ Excellent                    |
| Regulatory Acceptance | ✅ High                        | ⚠️ Needs Explainability           |
| Ease of Deployment    | ✅ Simple                      | ⚠️ More complex                   |
| Feature Engineering   | ⚠️ Needs manual binning        | ✅ Can auto-handle interactions   |

In regulated environments like banking, **interpretable models are often preferred** unless the gains from complex models are significant **and** explainability is added. A good practice is to build both, compare their business impact, and justify the choice based on both **regulatory** and **operational** KPIs.

---

_Sources:_

- Basel II Capital Accord
- HKMA Alternative Scoring
- World Bank Credit Scoring Guidelines
- TDS Scorecard Development
- Risk-Officer & CFI Readings
