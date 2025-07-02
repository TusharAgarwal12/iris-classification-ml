import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(iris)
print(df.head())

plt.scatter(df['petal_length'],df['petal_width'],c=df['species'].astype('category').cat.codes)
  #astype('category'): Converts text labels (strings) into a categorical data type
  # cat.codes: Assigns a unique number to each category: "setosa" → 0,"versicolor" → 1,"virginica" → 2
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title('Iris Species Scatter Plot')
plt.show()

X = df.drop('species', axis=1)  # Features
y = df['species']               # Target (labels)
""" df.drop('species', axis=1) means:
→ Drop (remove) the 'species' column
→ axis=1 means you're dropping a column, not a row """

# split the data(Train and test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""" 
test_size=0.2: 20% of the data will go to testing, 80% to training
random_state=42: Controls the random split — keeps it the same every time (for reproducibility) """


# Train the model  
model = LogisticRegression(max_iter=200)    # create the ml model
model.fit(X_train, y_train)    # Train the model using training data

#Predict and Evaluate
y_pred = model.predict(X_test)
print(y_pred)
     # Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
    # Detailed performance
print("Classification Report:\n", classification_report(y_test, y_pred))



































