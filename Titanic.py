import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Bước 1: Đọc và đánh giá dữ liệu
data = pd.read_csv("/mnt/data/titanic.csv")
print("Thông tin dữ liệu:")
print(data.info())
print("\nThống kê dữ liệu:")
print(data.describe())

# Bước 1.2: Kiểm tra giá trị bị thiếu
missing_values = data.isnull().sum()
print("\nGiá trị bị thiếu:")
print(missing_values[missing_values > 0])

# Bước 2: Điền giá trị bị thiếu
imputer = SimpleImputer(strategy='mean')
num_cols = data.select_dtypes(include=[np.number]).columns
data[num_cols] = imputer.fit_transform(data[num_cols])

# Bước 3: Chuyển đổi dữ liệu phân loại
cat_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Bước 4: Loại bỏ cột không cần thiết và chuẩn hóa
data.drop(columns=['Name'], inplace=True, errors='ignore')
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Bước 5: Chia tập dữ liệu
train, temp = train_test_split(data_scaled, test_size=0.3, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

X_train, y_train = train.drop(columns=['Survived']), train['Survived']
X_valid, y_valid = valid.drop(columns=['Survived']), valid['Survived']
X_test, y_test = test.drop(columns=['Survived']), test['Survived']

# Bước 6: Huấn luyện mô hình với MLflow
exp = mlflow.set_experiment("Titanic_Model_Training")
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train Linear Regression
with mlflow.start_run(experiment_id=exp.experiment_id):
    models["Linear Regression"].fit(X_train, y_train)
    y_pred_lr = models["Linear Regression"].predict(X_valid)
    mse_lr = mean_squared_error(y_valid, y_pred_lr)
    mlflow.log_metric("MSE_Linear_Regression", mse_lr)
    mlflow.sklearn.log_model(models["Linear Regression"], "linear_regression", input_example=X_valid.iloc[:1])
    
# Train Polynomial Regression
with mlflow.start_run(experiment_id=exp.experiment_id):
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_valid_poly = poly.transform(X_valid)
    models["Polynomial Regression"].fit(X_train_poly, y_train)
    y_pred_poly = models["Polynomial Regression"].predict(X_valid_poly)
    mse_poly = mean_squared_error(y_valid, y_pred_poly)
    mlflow.log_metric("MSE_Polynomial_Regression", mse_poly)
    mlflow.sklearn.log_model(models["Polynomial Regression"], "polynomial_regression", input_example=X_valid_poly[:1])
    
# Train Random Forest
with mlflow.start_run(experiment_id=exp.experiment_id):
    models["Random Forest"].fit(X_train, y_train)
    y_pred_rf = models["Random Forest"].predict(X_valid)
    mse_rf = mean_squared_error(y_valid, y_pred_rf)
    mlflow.log_metric("MSE_Random_Forest", mse_rf)
    mlflow.sklearn.log_model(models["Random Forest"], "random_forest", input_example=X_valid.iloc[:1])

# Bước 7: Dự đoán với mô hình Multiple Regression
with mlflow.start_run(experiment_id=exp.experiment_id):
    y_pred_lr_test = models["Linear Regression"].predict(X_test)
    mse_lr_test = mean_squared_error(y_test, y_pred_lr_test)
    print(f"MSE của Multiple Regression: {mse_lr_test}")
    mlflow.log_metric("MSE_Test_Linear_Regression", mse_lr_test)

# Bước 7: Dự đoán với mô hình Polynomial Regression
with mlflow.start_run(experiment_id=exp.experiment_id):
    X_test_poly = poly.transform(X_test)
    y_pred_poly_test = models["Polynomial Regression"].predict(X_test_poly)
    mse_poly_test = mean_squared_error(y_test, y_pred_poly_test)
    print(f"MSE của Polynomial Regression: {mse_poly_test}")
    mlflow.log_metric("MSE_Test_Polynomial_Regression", mse_poly_test)
