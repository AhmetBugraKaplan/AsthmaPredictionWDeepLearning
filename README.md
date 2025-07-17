# 🫁 Asthma Prediction – Deep Learning Based Health Data Analysis

This project uses a neural network to predict whether a person has asthma based on health and environmental features. Built using Python, TensorFlow, and scikit-learn, and trained/tested in Google Colab.

---

## 📊 Dataset Overview

- Demographic: Age, Gender, BMI, Smoking_Status  
- Medical: Family_History, Allergies, Comorbidities, Medication_Adherence  
- Environmental: Air_Pollution_Level, Physical_Activity_Level, Occupation_Type  
- Lung Function: Peak_Expiratory_Flow, FeNO_Level  
- Target: `Has_Asthma` (0 = No, 1 = Yes)

---

## 🧹 Data Preprocessing

- Missing values in `Allergies` and `Comorbidities` were filled.
- `get_dummies()` used for categorical encoding.
- Dropped ID and unused columns.
- Converted all columns to `float32` for neural network compatibility.

---

## 🤖 Model Architecture

```python
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=X.shape[1]))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
