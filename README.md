# 🩺 Chronic Kidney Disease Prediction

## 📌 Overview

A Machine Learning project that predicts whether a patient is likely to have **Chronic Kidney Disease (CKD)** based on medical and health parameters.

## 🎯 Objective

* Predict CKD using Machine Learning
* Analyze important health factors
* Compare different classification models
* Build an interactive prediction application

## 📊 Dataset

* **Records:** 397
* **Features:** 24+
* **Target:** CKD / Not CKD

The dataset contains patient information such as age, blood pressure, blood urea, serum creatinine, hemoglobin, diabetes, hypertension, and other medical parameters.

## 🧠 Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

## ⚙️ Workflow

**Data Collection → Preprocessing → EDA → Feature Engineering → Model Training → Evaluation → Prediction**

Models evaluated include:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* SVM

## 📈 Results

| Metric                    |      Score |
| ------------------------- | ---------: |
| **Best Model**            |   AdaBoost |
| Accuracy                  | **98.75%** |
| Precision                 | **98.77%** |
| Recall                    | **98.75%** |
| F1-Score                  | **98.75%** |
| Cross-Validation Accuracy | **97.78%** |

## 📂 Project Structure

```text
Kidney/
├── data/
├── notebooks/
├── src/
│   ├── components/
│   └── pipeline/
├── artifacts/
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🖥️ Run the Application

```bash
streamlit run app.py
```

## 🔍 Key Insights

* Serum creatinine and blood urea are important kidney-related indicators.
* Blood pressure, diabetes, and hypertension can be associated with CKD.
* AdaBoost achieved the best overall performance among the evaluated models.
