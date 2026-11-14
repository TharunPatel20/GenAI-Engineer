"""Project 02: ML and DL intro - train a loan approval classifier.

The script trains three models on the same loan data:
  1. Logistic regression (a simple, linear classifier)
  2. Decision tree with no depth limit (shows overfitting)
  3. Small neural network (MLP, a basic deep learning model)

Run:  python train_loan_model.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

SEED = 42  # Same seed each run, so results are repeatable.
DATA_FILE = Path(__file__).resolve().parent / "loan_data.csv"

# Step 1: Load the data
# Each row is one applicant. Loan_Status is the answer we want to predict (labeled data).
data = pd.read_csv(DATA_FILE)
print("First 5 rows:")
print(data.head())
print(f"\nRows: {len(data)}")

# Step 2: Make the number columns smaller and encode text columns as numbers
# Income and loan amounts are shown in thousands (for example 6065 becomes 6.065).
# Smaller numbers help the models train more smoothly.
for col in ["Applicant_Income", "Coapplicant_Income", "Loan_Amount"]:
    data[col] = data[col] / 1000
# Models only understand numbers, so text categories become 0/1 values.
data["Education"] = data["Education"].map({"Not Graduate": 0, "Graduate": 1})
data["Loan_Status"] = data["Loan_Status"].map({"N": 0, "Y": 1})
# One-hot encoding: Property_Area becomes one 0/1 column per area.
data = pd.get_dummies(data, columns=["Property_Area"], dtype=int)

# Step 3: Split into inputs (X) and the target (y)
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Step 4: Split into training (80%) and testing (20%) sets
# The model learns from the training rows. The test rows are held back to check it on unseen data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(f"\nTraining rows: {len(X_train)}   Testing rows: {len(X_test)}")

# Step 5: Train logistic regression and measure accuracy
# Accuracy = share of correct predictions (0.80 means 80 out of 100 are right).
print("\n--- Model 1: Logistic regression ---")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print(f"Train accuracy: {logreg.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {logreg.score(X_test, y_test):.3f}")

# Step 6: Show overfitting with a decision tree that has no depth limit
# The tree keeps splitting until it memorizes the training rows.
# Expect a high train score and a clearly lower test score.
print("\n--- Model 2: Decision tree with no depth limit (overfitting demo) ---")
tree = DecisionTreeClassifier(random_state=SEED)  # max_depth not set = no limit
tree.fit(X_train, y_train)
train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Gap (train minus test): {train_acc - test_acc:.3f}")

# Step 7: Train a small neural network (MLP) and compare
# One hidden layer with 8 neurons. ReLU is the activation function inside the hidden layer.
# Training repeats forward and backward propagation (epochs) until it stops improving.
print("\n--- Model 3: Small neural network (MLP, one hidden layer of 8 neurons) ---")
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    max_iter=2000,
    random_state=SEED,
)
mlp.fit(X_train, y_train)
print(f"Train accuracy: {mlp.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {mlp.score(X_test, y_test):.3f}")

# Step 8: Compare the three models on the test set
print("\n--- Summary (test accuracy) ---")
print(f"Logistic regression: {logreg.score(X_test, y_test):.3f}")
print(f"Decision tree:       {test_acc:.3f}")
print(f"Neural network:      {mlp.score(X_test, y_test):.3f}")
