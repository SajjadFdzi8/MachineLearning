# -*- coding: utf-8 -*-
"""Question_2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V8EERy7gCFzySTVCXGKej3mzc4LucKU_
"""

!pip install gdown
file_url = "https://drive.google.com/uc?id=180FkupJQe0Oiq0A1yJg4m5Ggbfw-G8Sc"
!gdown {file_url} -O data.npy

import numpy as np

data = np.load("data.npy")
print(data)

data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

y_data = np.load("data.npy")
x_data = np.linspace(-6, 13, len(y_data))
data_df = pd.DataFrame({'x': x_data, 'y': y_data})

sns.pairplot(data_df)
plt.show()

test_size = int(0.15 * len(data_df))
test_indices = np.random.choice(data_df.index, size=test_size, replace=False)
test_data = data_df.loc[test_indices]
train_data = data_df.drop(test_indices)

plt.figure(figsize=(10, 6))
plt.scatter(train_data['x'], train_data['y'], color='blue', label='Train Data', alpha=0.7)
plt.scatter(test_data['x'], test_data['y'], color='red', label='Test Data', alpha=0.7)

plt.title("Random Train vs Test Data", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

x_train = train_data['x'].values
y_train = train_data['y'].values
x_test = test_data['x'].values
y_test = test_data['y'].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train.reshape(-1, 1)).flatten()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
x_test_scaled = scaler_x.transform(x_test.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

x_mean_scaled = np.mean(x_train_scaled)
y_mean_scaled = np.mean(y_train_scaled)
numerator = np.sum((x_train_scaled - x_mean_scaled) * (y_train_scaled - y_mean_scaled))
denominator = np.sum((x_train_scaled - x_mean_scaled) ** 2)
w_scaled = numerator / denominator
b_scaled = y_mean_scaled - w_scaled * x_mean_scaled

y_train_pred_scaled = w_scaled * x_train_scaled + b_scaled
y_test_pred_scaled = w_scaled * x_test_scaled + b_scaled

y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Train Data', alpha=0.7)
plt.scatter(x_test, y_test, color='red', label='Test Data', alpha=0.7)
plt.plot(x_train, y_train_pred, color='green', label='Regression Line', linewidth=2)
plt.title("Linear Regression with Normalized Data", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

print(w_scaled)
print(b_scaled)

y_train_pred_scaled = w_scaled * x_train_scaled + b_scaled
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()

train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

y_test_pred_scaled = w_scaled * x_test_scaled + b_scaled
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

plt.figure(figsize=(10, 6))
plt.bar(['Train', 'Test'], [train_mse, test_mse], color=['blue', 'red'])
plt.title('Mean Squared Error (MSE)', fontsize=16)
plt.ylabel('MSE', fontsize=14)
plt.xlabel('Dataset', fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(['Train', 'Test'], [train_mae, test_mae], color=['blue', 'red'])
plt.title('Mean Absolute Error (MAE)', fontsize=16)
plt.ylabel('MAE', fontsize=14)
plt.xlabel('Dataset', fontsize=14)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(['Train', 'Test'], [train_r2, test_r2], color=['blue', 'red'])
plt.title('R-squared', fontsize=16)
plt.ylabel('R-squared', fontsize=14)
plt.xlabel('Dataset', fontsize=14)
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()

train_sizes = []
train_mse = []
train_mae = []
train_r2 = []

test_mse = []
test_mae = []
test_r2 = []

for i in range(1, len(x_train_scaled) + 1):
    x_train_subset = x_train_scaled[:i].reshape(-1, 1)
    y_train_subset = y_train_scaled[:i]

    model.fit(x_train_subset, y_train_subset)

    y_train_subset_pred = model.predict(x_train_subset)
    train_mse.append(mean_squared_error(y_train_subset, y_train_subset_pred))
    train_mae.append(mean_absolute_error(y_train_subset, y_train_subset_pred))
    train_r2.append(r2_score(y_train_subset, y_train_subset_pred))

    y_test_pred_scaled = model.predict(x_test_scaled.reshape(-1, 1))
    test_mse.append(mean_squared_error(y_test_scaled, y_test_pred_scaled))
    test_mae.append(mean_absolute_error(y_test_scaled, y_test_pred_scaled))
    test_r2.append(r2_score(y_test_scaled, y_test_pred_scaled))

    train_sizes.append(i)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mse, label='Train MSE', color='blue', marker='o')
plt.plot(train_sizes, test_mse, label='Test MSE', color='red', marker='o')
plt.title('Mean Squared Error (MSE)', fontsize=16)
plt.xlabel('Number of Training Samples', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mae, label='Train MAE', color='blue', marker='o')
plt.plot(train_sizes, test_mae, label='Test MAE', color='red', marker='o')
plt.title('Mean Absolute Error (MAE)', fontsize=16)
plt.xlabel('Number of Training Samples', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_r2, label='Train R-squared', color='blue', marker='o')
plt.plot(train_sizes, test_r2, label='Test R-squared', color='red', marker='o')
plt.title('R-squared', fontsize=16)
plt.xlabel('Number of Training Samples', fontsize=14)
plt.ylabel('R-squared', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

metrics = {"Degree": [], "MSE_Train": [], "MSE_Test": [], "MAE_Train": [], "MAE_Test": [], "R-squared_Train": [], "R-squared_Test": []}

for degree in degrees:
    poly = PolynomialFeatures(degree)
    x_train_poly = poly.fit_transform(x_train_scaled.reshape(-1, 1))
    x_test_poly = poly.transform(x_test_scaled.reshape(-1, 1))

    coef = np.linalg.pinv(x_train_poly) @ y_train_scaled
    y_train_pred_scaled = x_train_poly @ coef
    y_test_pred_scaled = x_test_poly @ coef
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    metrics["Degree"].append(degree)
    metrics["MSE_Train"].append(mse_train)
    metrics["MSE_Test"].append(mse_test)
    metrics["MAE_Train"].append(mae_train)
    metrics["MAE_Test"].append(mae_test)
    metrics["R-squared_Train"].append(r2_train)
    metrics["R-squared_Test"].append(r2_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train_scaled, y_train_scaled, color='blue', label='Train Data', alpha=0.5)
    plt.scatter(x_test_scaled, y_test_scaled, color='red', label='Test Data', alpha=0.5)
    x_range = np.linspace(x_train_scaled.min(), x_train_scaled.max(), 500).reshape(-1, 1)
    y_range_pred_scaled = poly.transform(x_range) @ coef
    plt.plot(x_range, y_range_pred_scaled, color='green', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}', fontsize=14)
    plt.legend()
    plt.xlabel('Scaled X')
    plt.ylabel('Scaled Y')
    plt.show()

metrics_df = pd.DataFrame(metrics)
metrics_df.set_index("Degree", inplace=True)

plt.figure(figsize=(12, 6))
metrics_df[['MSE_Train', 'MSE_Test']].plot(marker='o', title="MSE by Polynomial Degree", grid=True)
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.show()

plt.figure(figsize=(12, 6))
metrics_df[['MAE_Train', 'MAE_Test']].plot(marker='o', title="MAE by Polynomial Degree", grid=True)
plt.xlabel("Polynomial Degree")
plt.ylabel("MAE")
plt.show()

plt.figure(figsize=(12, 6))
metrics_df[['R-squared_Train', 'R-squared_Test']].plot(marker='o', title="R-squared by Polynomial Degree", grid=True)
plt.xlabel("Polynomial Degree")
plt.ylabel("R-squared")
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

all_models = {
    "Decision Tree": DecisionTreeRegressor(random_state=93),
    "Support Vector Regressor": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=93),
}

metrics = {"Train": {"MSE": [], "MAE": [], "R2": []}, "Test": {"MSE": [], "MAE": [], "R2": []}}
predictions = {}

for name, model in all_models.items():
    model.fit(x_train_scaled.reshape(-1, 1), y_train_scaled)

    y_train_pred_scaled = model.predict(x_train_scaled.reshape(-1, 1))
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()

    y_test_pred_scaled = model.predict(x_test_scaled.reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    predictions[name] = {"Train": y_train_pred, "Test": y_test_pred}

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics["Train"]["MSE"].append(train_mse)
    metrics["Train"]["MAE"].append(train_mae)
    metrics["Train"]["R2"].append(train_r2)

    metrics["Test"]["MSE"].append(test_mse)
    metrics["Test"]["MAE"].append(test_mae)
    metrics["Test"]["R2"].append(test_r2)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train_scaled, y_train_scaled, color='blue', label='Train Data', alpha=0.5)
    plt.scatter(x_test_scaled, y_test_scaled, color='red', label='Test Data', alpha=0.5)
    x_range = np.linspace(x_train_scaled.min(), x_train_scaled.max(), 500).reshape(-1, 1)
    y_range_pred_scaled = model.predict(x_range)
    plt.plot(x_range, y_range_pred_scaled, color='green', label=f'Regression Line')
    plt.title(f"{name} Regression", fontsize=14)
    plt.xlabel("Normalized x")
    plt.ylabel("Normalized y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

metrics_df = pd.DataFrame(metrics)

for criterion in ["MSE", "MAE", "R2"]:
    plt.figure(figsize=(12, 6))
    train_values = metrics["Train"][criterion]
    test_values = metrics["Test"][criterion]
    x_labels = list(all_models.keys())

    plt.bar([i - 0.2 for i in range(len(x_labels))], train_values, width=0.4, label="Train", color="blue", alpha=0.7)
    plt.bar([i + 0.2 for i in range(len(x_labels))], test_values, width=0.4, label="Test", color="red", alpha=0.7)

    plt.xticks(range(len(x_labels)), x_labels, rotation=15)
    plt.title(f"{criterion} for Train and Test Data", fontsize=14)
    plt.ylabel(criterion, fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

max_degree = 6
lambda_param = 1000

train_mse_list = []
test_mse_list = []
train_mae_list = []
test_mae_list = []
train_r2_list = []
test_r2_list = []

for degree in range(1, max_degree + 1):
    X_poly_train = np.column_stack([x_train_scaled ** i for i in range(1, degree + 1)])
    X_poly_test = np.column_stack([x_test_scaled ** i for i in range(1, degree + 1)])

    X_mean = np.mean(X_poly_train, axis=0)
    Y_mean = np.mean(y_train_scaled)
    X_centered = X_poly_train - X_mean
    Y_centered = y_train_scaled - Y_mean

    regularization_term = lambda_param * np.eye(X_poly_train.shape[1])
    w = np.linalg.inv(X_centered.T @ X_centered + regularization_term) @ X_centered.T @ Y_centered
    b = Y_mean - np.sum(w * X_mean)

    y_train_pred_scaled = X_poly_train @ w + b
    y_test_pred_scaled = X_poly_test @ w + b
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    train_mae_list.append(train_mae)
    test_mae_list.append(test_mae)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)

    print(f"Degree {degree}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")

degrees = list(range(1, max_degree + 1))

plt.figure(figsize=(12, 8))
plt.plot(degrees, train_mse_list, label='Train MSE', marker='o', color='blue')
plt.plot(degrees, test_mse_list, label='Test MSE', marker='o', color='red')
plt.title("Mean Squared Error (MSE) vs Degree with Regularization", fontsize=16)
plt.xlabel("Degree", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(degrees, train_mae_list, label='Train MAE', marker='o', color='blue')
plt.plot(degrees, test_mae_list, label='Test MAE', marker='o', color='red')
plt.title("Mean Absolute Error (MAE) vs Degree with Regularization", fontsize=16)
plt.xlabel("Degree", fontsize=14)
plt.ylabel("MAE", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(degrees, train_r2_list, label='Train R-squared', marker='o', color='blue')
plt.plot(degrees, test_r2_list, label='Test R-squared', marker='o', color='red')
plt.title("R-squared vs Degree with Regularization", fontsize=16)
plt.xlabel("Degree", fontsize=14)
plt.ylabel("R-squared", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()