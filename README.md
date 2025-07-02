# 🌸 Iris Flower Classification using Logistic Regression

This project is part of my Machine Learning learning path — where I built an ML model to classify Iris flower species using the Iris dataset from a public GitHub source.

## 🚀 Project Overview

- Built a Logistic Regression model to classify flowers into:
  - Setosa
  - Versicolor
  - Virginica
- Achieved high accuracy with minimal preprocessing
- Visualized feature relationships using Matplotlib
- Evaluated model using accuracy and classification report

---

## 📂 Dataset

- Source: [Seaborn GitHub Repository](https://github.com/mwaskom/seaborn-data/blob/master/iris.csv)
- Features:
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
- Target: `species`

---

## 🧠 Technologies Used

- Python
- Pandas & NumPy
- Matplotlib
- Scikit-learn (LogisticRegression)

---

## 📈 Steps Performed

1. Loaded dataset using GitHub CSV link via Pandas
2. Explored and visualized the data
3. Split into training and testing data
4. Trained a Logistic Regression model
5. Evaluated model using accuracy and classification report
6. Made predictions on new flower data

---

## 🔍 Sample Code Snippet

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
