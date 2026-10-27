
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. DATASET CREATION
# ==========================================

np.random.seed(42)

# 500 students
study_hours = np.random.uniform(0, 10, 500)
attendance = np.random.uniform(30, 100, 500)

# Create target labels
# 1 = Pass, 0 = Fail
score = study_hours * 6 + attendance * 0.4

y = (score >= 55).astype(int)

# Input features
X = np.column_stack((study_hours, attendance))

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# 2. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert target to column vector
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# ==========================================
# 3. DATA NORMALIZATION
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 4. ACTIVATION FUNCTIONS
# ==========================================

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ==========================================
# 5. ANN MODEL INITIALIZATION
# ==========================================

input_size = 2
hidden1_size = 8
hidden2_size = 4
output_size = 1

# He initialization
W1 = np.random.randn(
    input_size, hidden1_size
) * np.sqrt(2 / input_size)

b1 = np.zeros((1, hidden1_size))

W2 = np.random.randn(
    hidden1_size, hidden2_size
) * np.sqrt(2 / hidden1_size)

b2 = np.zeros((1, hidden2_size))

W3 = np.random.randn(
    hidden2_size, output_size
) * np.sqrt(2 / hidden2_size)

b3 = np.zeros((1, output_size))


# ==========================================
# 6. HYPERPARAMETERS
# ==========================================

learning_rate = 0.001
epochs = 1000
batch_size = 32

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

m = {}
v = {}

parameters = {
    "W1": W1, "b1": b1,
    "W2": W2, "b2": b2,
    "W3": W3, "b3": b3
}

for key, param in parameters.items():
    m[key] = np.zeros_like(param)
    v[key] = np.zeros_like(param)


# ==========================================
# 7. FORWARD PROPAGATION
# ==========================================

def forward(X):

    # Hidden Layer 1
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Hidden Layer 2
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    # Output Layer
    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)

    cache = (X, Z1, A1, Z2, A2, Z3, A3)

    return A3, cache


# ==========================================
# 8. LOSS FUNCTION
# ==========================================

def binary_cross_entropy(y_true, y_pred):

    y_pred = np.clip(
        y_pred, 1e-8, 1 - 1e-8
    )

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# ==========================================
# 9. BACKPROPAGATION
# ==========================================

def backward(y, cache):

    X, Z1, A1, Z2, A2, Z3, A3 = cache

    n = X.shape[0]

    # Output Layer
    dZ3 = (A3 - y) / n

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    # Hidden Layer 2
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden Layer 1
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2,
        "W3": dW3, "b3": db3
    }

    return gradients


# ==========================================
# 10. ADAM OPTIMIZER
# ==========================================

def adam_update(gradients, t):

    global W1, b1, W2, b2, W3, b3

    for key in parameters:

        g = gradients[key]

        # First moment
        m[key] = beta1 * m[key] + (1 - beta1) * g

        # Second moment
        v[key] = beta2 * v[key] + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m[key] / (1 - beta1 ** t)
        v_hat = v[key] / (1 - beta2 ** t)

        # Parameter update
        parameters[key] -= (
            learning_rate *
            m_hat /
            (np.sqrt(v_hat) + epsilon)
        )


# ==========================================
# 11. TRAINING LOOP
# ==========================================

train_losses = []
test_losses = []

step = 0

for epoch in range(epochs):

    # Shuffle training data
    indices = np.random.permutation(len(X_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward
        predictions, cache = forward(X_batch)

        # Backward
        gradients = backward(y_batch, cache)

        # Update
        step += 1
        adam_update(gradients, step)

    # Calculate epoch losses
    train_pred, _ = forward(X_train)
    test_pred, _ = forward(X_test)

    train_loss = binary_cross_entropy(
        y_train, train_pred
    )

    test_loss = binary_cross_entropy(
        y_test, test_pred
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f}"
        )


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

test_probabilities, _ = forward(X_test)

y_pred = (test_probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy :", accuracy * 100, "%")
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 13. LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training Performance")

plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 14. NEW DATA PREDICTION
# ==========================================

# New student:
# Study Hours = 7
# Attendance = 85

new_student = np.array([[7, 85]])

# Apply same scaler
new_student = scaler.transform(new_student)

# Prediction
probability, _ = forward(new_student)

predicted_class = (
    probability >= 0.5
).astype(int)

print("\n========== NEW STUDENT PREDICTION ==========")

print("Pass Probability:", probability[0][0])

if predicted_class[0][0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
