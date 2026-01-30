# 1. Import Required Libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 2. Load the Dataset

df = pd.read_csv("loan_prediction_dataset.csv")

# Remove extra spaces in column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)
print(df.head())

# 3. Separate Features and Target

X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# 4. Encode Target Column

le = LabelEncoder()
y = le.fit_transform(y)
# Approved -> 1, Rejected -> 0

# 5. One-Hot Encode Categorical Features

X = pd.get_dummies(X, columns=["Employment_Status"], drop_first=True)

print("\nEncoded Feature Columns:")
print(X.columns)

# 6. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 7. Build Decision Tree Model

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

# 8. Train the Model

model.fit(X_train, y_train)

# 9. Make Predictions

y_pred = model.predict(X_test)

# 10. Evaluate Model

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

# 11. Visualize Decision Tree

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(cls) for cls in le.classes_],
    filled=True
)
plt.show()
