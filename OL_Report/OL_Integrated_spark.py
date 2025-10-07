# Install missing packages
%pip install seaborn

# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from time import time

# --- Step 0: One-time Setup & Parity Guarantees ---
# Fix seeds for reproducibility (Python, NumPy, and PyTorch)
def set_seed(s=4242):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4242)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Step 1: Data Handling ---
# Load datasets
hotel_data = pd.read_csv('hotel_bookings.csv')
us_accidents_data = pd.read_csv('US_Accidents_March23.csv')

# Preprocess datasets
def preprocess_hotel_booking(data):
    """
    Preprocess the Hotel Booking dataset.
    """
    # Remove post-outcome fields to prevent leakage
    data = data.drop(columns=['reservation_status'])

    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Impute missing values for numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # Impute missing values for categorical columns using the most frequent strategy
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    # Encode categorical variables
    encoder = TargetEncoder()
    # Ensure the target variable is passed to the encoder
    data[categorical_cols] = encoder.fit_transform(data[categorical_cols], data['is_canceled'])

    # Scale numeric features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data


def preprocess_us_accidents(data):
    """
    Preprocess the US Accidents dataset.
    """
    # Clean the datetime columns to remove fractional seconds or extraneous characters
    data['End_Time'] = data['End_Time'].str.split('.').str[0]
    data['Start_Time'] = data['Start_Time'].str.split('.').str[0]

    # Calculate the duration in minutes
    data['Duration'] = (pd.to_datetime(data['End_Time'], format='mixed') -
                       pd.to_datetime(data['Start_Time'], format='mixed')).dt.total_seconds() / 60

    # Scale numeric features
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data


hotel_data_preprocessed = preprocess_hotel_booking(hotel_data)
us_accidents_data_preprocessed = preprocess_us_accidents(us_accidents_data)

# Split datasets
X_hotel_train, X_hotel_test, y_hotel_train, y_hotel_test = train_test_split(
    hotel_data_preprocessed.drop('is_canceled', axis=1),
    hotel_data_preprocessed['is_canceled'],
    test_size=0.2,
    random_state=42
)
X_accidents_train, X_accidents_test, y_accidents_train, y_accidents_test = train_test_split(
    us_accidents_data_preprocessed.drop('Duration', axis=1),
    us_accidents_data_preprocessed['Duration'],
    test_size=0.2,
    random_state=42
)

# Inspect the data types of the columns
print(X_accidents_train.dtypes)

# Drop non-numeric columns (if they are not needed)
X_accidents_train = X_accidents_train.select_dtypes(include=['float64', 'int64'])

# Encode categorical columns (if any)
categorical_cols = X_accidents_train.select_dtypes(include=['object']).columns
if not categorical_cols.empty:
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_accidents_train[categorical_cols] = encoder.fit_transform(X_accidents_train[categorical_cols])

# Impute missing values (if any)
imputer = SimpleImputer(strategy='most_frequent')
X_accidents_train = pd.DataFrame(imputer.fit_transform(X_accidents_train), columns=X_accidents_train.columns)

# Convert to numeric types
X_accidents_train = X_accidents_train.astype(float)

# Validate the data types
print(X_accidents_train.dtypes)

# Convert to PyTorch tensors
Xva = torch.from_numpy(X_hotel_test.values).float()
yva = torch.from_numpy(y_hotel_test.values).long()
Xte = torch.from_numpy(X_accidents_train.values).float()
yte = torch.from_numpy(y_accidents_train.values).float()

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True, drop_last=False)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=1024, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=1024, shuffle=False)

# --- Step 2: Model Definition ---
# Mirror the sklearn MLP with an nn.Module
# Use the SAME hidden sizes and activations you claimed in SL; that’s your “fixed backbone”
# Keep output head/activation appropriate for the task (e.g., CrossEntropyLoss for multiclass, MSELoss for regression)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[128, 64], out_dim=4, dropout_p=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
            if dropout_p > 0:
                layers += [nn.Dropout(p=dropout_p)]
        layers += [nn.Linear(hidden[-1] if hidden else in_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Example usage
model = MLP(in_dim=Xtr.shape[1], hidden=[128, 64], out_dim=num_classes).to(device)

# --- Step 3: Freezing Layers ---
# Freeze all but the last k layers (this is the crux for RO and for Part-specific constraints)
def linear_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]

def freeze_all_but_last_k(model, k=2):
    layers = linear_layers(model)
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze last k Linear layers
    for m in layers[-k:]:
        for p in m.parameters():
            p.requires_grad = True
    # Report counts (needed for RO cap)
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total={tot:,} | trainable(last {k})={trainable:,}")
    return trainable

# Example: freeze all but last 1–3 layers for Part 1 RO
trainable = freeze_all_but_last_k(model, k=2)
assert trainable <= 50_000, "RO parameter cap exceeded."

# --- Step 4: Losses & Metrics ---
# Define appropriate loss functions and metrics based on the task (classification or regression)
task = "classification"  # or "regression"
criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

# Add task-appropriate metrics (Accuracy/F1/AUROC vs. MAE/MSE/R2) consistent with your SL choices
def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification performance using ROC-AUC, PR-AUC, and F1-Score.
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    return roc_auc, pr_auc, f1

def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression performance using MAE and MSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mse

# --- Step 5: Training and Evaluation Loop ---
# Implement a minimal, transparent training/eval loop (counts “gradient evaluations” cleanly)
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n, grad_evals = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(xb)
            loss = criterion(pred, yb)
            if is_train:
                loss.backward()
                optimizer.step()
                grad_evals += 1  # count one optimizer step = one gradient evaluation
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n, grad_evals

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n

# --- Step 6: Optimizer Ablations (Part 2) ---
# Define the optimizers exactly as the assignment lists; keep everything else fixed (batch size, schedule form, seeds)
# Record time/steps-to-ℓ and stability over seeds
# Do not call AdamW “Adam baseline”

def make_opt(model, kind, lr, **kwargs):
    if kind == "sgd":
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if kind == "momentum":
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=kwargs.get("momentum", 0.9))
    if kind == "nesterov":
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=kwargs.get("momentum", 0.9), nesterov=True)
    if kind == "adam":
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(kwargs.get("beta1", 0.9), kwargs.get("beta2", 0.999)), eps=kwargs.get("eps", 1e-8))
    if kind == "adamw":
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(kwargs.get("beta1", 0.9), kwargs.get("beta2", 0.999)), eps=kwargs.get("eps", 1e-8), weight_decay=kwargs.get("wd", 1e-2))
    raise ValueError(kind)

# --- Step 7: Freezing and RO Hygiene (Part 1) ---
# For RO, call model.eval() for every objective (dropout off; BN uses stored stats)
# Freeze all but last 1–3 layers (≤ ~50k params)
# Define the objective as full-validation loss
# Count one function evaluation per full validation pass
# Do not interleave gradient steps in RO

@torch.no_grad()
def validation_objective(model, val_loader):
    model.eval()
    return evaluate(model, val_loader)  # one full pass = 1 function evaluation

# --- Step 8: Regularization Study (Part 3) ---
# Keep Adam hyperparams fixed to the best from Part 2 (no switching to AdamW; no retuning LR when adding regularization)
# Implement L2 (coupled) via loss term (not AdamW), early stopping rule, dropout placements (document where), label smoothing or target noise, and modest augmentation appropriate to the modality (off for val/test)
# Budget-match runs and report dispersion across seeds

def run_epoch_with_l2(model, loader, optimizer, l2_lambda=0.0):
    model.train()
    total_loss, n, grad_evals = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        if l2_lambda > 0:
            l2 = sum((p**2).sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l2_lambda * l2
        loss.backward()
        optimizer.step()
        grad_evals += 1
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n, grad_evals

# --- Step 9: Reporting & Accounting ---
# Compute accounting: gradient evals (updates), function evals (RO), wall-clock on the same hardware class
# Threshold ℓ once per dataset; show steps/time to ℓ; include failures as “> budget”

# Define the train_to_budget function
def train_to_budget(model, optimizer, train_loader, val_loader, max_updates=10000, L_threshold=None):
    """
    Train a model with a given optimizer and dataset loaders within a specified budget of gradient evaluations.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        max_updates (int): Maximum number of gradient evaluations (updates) allowed.
        L_threshold (float, optional): Validation loss threshold to stop training early.

    Returns:
        dict: A dictionary containing the best validation loss, total gradient evaluations, training time, and whether the threshold was reached.
    """
    grad_evals_total, best_val, t0 = 0, float("inf"), time()
    reached = None

    while grad_evals_total < max_updates:
        # Train for one epoch
        tr_loss, ge = run_epoch(model, train_loader, optimizer)
        grad_evals_total += ge

        # Evaluate on the validation set
        val_loss = evaluate(model, val_loader)
        best_val = min(best_val, val_loss)

        # Check if the validation loss threshold is met
        if L_threshold is not None and reached is None and val_loss <= L_threshold:
            reached = (grad_evals_total, time() - t0)

    return {
        "best_val": best_val,
        "grad_evals": grad_evals_total,
        "time_sec": time() - t0,
        "reached_L": reached
    }

# Example usage
optimizer = make_opt(model, "adam", lr=0.001)
results = train_to_budget(model, optimizer, train_loader, val_loader, max_updates=10000)
print(results)

# --- Step 10: SL Code Integration ---
# Train and evaluate models using sklearn pipelines
def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier with hyperparameter tuning.
    """
    param_grid = {
        'max_depth': [8, 16],
        'min_samples_leaf': [100, 200]
    }
    tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(tree, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def train_shallow_nn(X_train, y_train, input_dim):
    """
    Train a shallow neural network with SGD optimizer.
    """
    class ShallowNN(nn.Module):
        def __init__(self, input_dim):
            super(ShallowNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    model = ShallowNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

def train_deep_nn(X_train, y_train, input_dim):
    """
    Train a deep neural network with SGD optimizer.
    """
    class DeepNN(nn.Module):
        def __init__(self, input_dim):
            super(DeepNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 128)
            self.fc5 = nn.Linear(128, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.sigmoid(self.fc5(x))
            return x

    model = DeepNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

# Train models
decision_tree_model = train_decision_tree(X_hotel_train, y_hotel_train)
knn_model = train_knn(X_hotel_train, y_hotel_train)
svm_model = train_svm(X_hotel_train, y_hotel_train)
shallow_nn_model = train_shallow_nn(X_hotel_train, y_hotel_train, input_dim=X_hotel_train.shape[1])
deep_nn_model = train_deep_nn(X_hotel_train, y_hotel_train, input_dim=X_hotel_train.shape[1])

# Evaluate models
roc_auc, pr_auc, f1 = evaluate_classification(y_hotel_test, decision_tree_model.predict(X_hotel_test))
print(f"Decision Tree - ROC-AUC: {roc_auc}, PR-AUC: {pr_auc}, F1-Score: {f1}")

# Plot learning curves
plot_learning_curve(decision_tree_model, X_hotel_train, y_hotel_train, 'Learning Curve for Decision Tree')
plot_residuals(y_accidents_test, svm_model.predict(X_accidents_test), 'Residuals for SVM Regressor')

# Randomized Optimization
model = ShallowNN(input_dim=X_hotel_train.shape[1])
best_state_rhc, best_fitness_rhc = rhc_optimizer(model, objective_function, max_iterations=1000)
best_state_sa, best_fitness_sa = sa_optimizer(model, objective_function, max_iterations=1000)
best_state_ga, best_fitness_ga = ga_optimizer(model, objective_function, max_iterations=1000)

# Adam Variants Comparison
optimizers = [
    torch.optim.SGD(model.parameters(), lr=0.01),
    torch.optim.Adam(model.parameters(), lr=0.001),
    torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08),
    torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
]

for optimizer in optimizers:
    training_time = train_model_with_optimizer(model, optimizer, nn.BCELoss(), dataloader, num_epochs=15)
    print(f"Training time with {optimizer}: {training_time}")

# Regularization Techniques Evaluation
regularization_techniques = ["L2", "Dropout", "Label Smoothing", "Data Augmentation"]
results = evaluate_regularization(model, dataloader, nn.BCELoss(), regularization_techniques)
print(f"Regularization results: {results}")

# Integrated Approach
integrated_training_time = integrate_best_combination(model, dataloader, nn.BCELoss())
print(f"Integrated training time: {integrated_training_time}")
