# import pandas as pd
# import streamlit as st
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# import numpy as np


# def main():
#     # Đọc tập dữ liệu Titanic
#     df = pd.read_csv('titanic.csv')

#     st.title("Hướng Dẫn Tiền Xử Lý Dữ Liệu Titanic")

#     st.header("1. Xem Dữ Liệu Ban Đầu")
#     st.write("Dữ liệu gốc trước khi tiền xử lý:")
#     st.dataframe(df)

#     st.header("2. Kiểm Tra Giá Trị Thiếu")
#     st.write("Thông tin dữ liệu và giá trị bị thiếu:")
#     st.text(df.info())
#     st.text(df.isnull().sum())

#     # Xử lý dữ liệu: điền giá trị thiếu tùy theo từng cột
#     for column in df.columns:
#         if df[column].dtype == 'object':
#             df[column].fillna(df[column].mode()[0], inplace=True)
#         else:
#             df[column].fillna(df[column].median(), inplace=True)

#     st.header("3. Điền Giá Trị Thiếu")
#     st.write("Dữ liệu sau khi điền giá trị thiếu:")
#     st.dataframe(df)

#     # Chuyển đổi dữ liệu phân loại thành số nguyên tuần tự
#     label_encoders = {}
#     categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket']

#     for col in categorical_columns:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le

#     st.header("4. Chuyển Đổi Dữ Liệu Phân Loại")
#     st.write("Dữ liệu sau khi mã hóa các biến phân loại thành số:")
#     st.dataframe(df)

#     # Chuẩn hóa các cột số (loại bỏ cột 'Name' nếu tồn tại)
#     if 'Name' in df.columns:
#         df.drop(columns=['Name'], inplace=True)

#     scaler = StandardScaler()
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#     st.header("5. Chuẩn Hóa Dữ Liệu")
#     st.write("Dữ liệu sau khi chuẩn hóa:")
#     st.dataframe(df)

#     # Người dùng chọn tỷ lệ train/validation, giữ nguyên tập test 15%
#     train_size = st.slider("Chọn tỷ lệ tập train (%)", 50, 85, 70) / 100
#     valid_size = 1 - train_size

#     temp_data, test_data = train_test_split(df, test_size=0.15, random_state=42)
#     train_data, valid_data = train_test_split(temp_data, test_size=valid_size/(train_size + valid_size), random_state=42)

#     st.header("6. Chia Dữ Liệu")
#     st.write(f'Train size: {train_data.shape}, Validation size: {valid_data.shape}, Test size: {test_data.shape}')
#     st.write("Dữ liệu sau khi chia:")
#     st.dataframe(train_data)

#     # Lựa chọn mô hình
#     st.header("7. Huấn Luyện Mô Hình")
#     model_choice = st.selectbox("Chọn mô hình học máy", ["Multiple Regression", "Polynomial Regression", "Random Forest"])
#     cv_folds = st.slider("Chọn số lượng folds cho Cross Validation", 2, 20, 10)

#     X_train = train_data.drop(columns=['Survived'])
#     y_train = train_data['Survived']
#     X_valid = valid_data.drop(columns=['Survived'])
#     y_valid = valid_data['Survived']

#     if model_choice == "Multiple Regression":
#         model = Ridge(alpha=1.0)
#         param_grid = {'alpha': [0.1, 1.0, 10.0]}
#         grid_search = GridSearchCV(model, param_grid, cv=cv_folds)
#         grid_search.fit(X_train, y_train)
#         model = grid_search.best_estimator_
#     elif model_choice == "Polynomial Regression":
#         degree = st.slider("Chọn bậc của đa thức", 2, 5, 2)
#         model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
#         param_grid = {'ridge__alpha': [0.1, 1.0, 10.0]}
#         grid_search = GridSearchCV(model, param_grid, cv=cv_folds)
#         grid_search.fit(X_train, y_train)
#         model = grid_search.best_estimator_
#     elif model_choice == "Random Forest":
#         model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
#         model.fit(X_train, y_train)

#     # Đánh giá mô hình
#     train_score = model.score(X_train, y_train)
#     valid_score = model.score(X_valid, y_valid)

#     st.write(f"Độ chính xác trên tập huấn luyện: {train_score:.4f}")
#     st.write(f"Độ chính xác trên tập validation: {valid_score:.4f}")

# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Streamlit UI
st.title("Dự đoán Titanic Survival")
def main():
    # Bước 1: Đọc và đánh giá dữ liệu
    data = pd.read_csv("titanic.csv")
    st.write("### Dữ liệu ban đầu:")
    st.dataframe(data)

    # Bước 1.2: Kiểm tra giá trị bị thiếu
    missing_values = data.isnull().sum()
    st.write("### Giá trị bị thiếu:")
    st.write(missing_values[missing_values > 0])

    # Bước 2: Điền giá trị bị thiếu
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    st.write("### Dữ liệu sau khi điền giá trị thiếu:")
    st.dataframe(data)

    # Bước 3: Chuyển đổi dữ liệu phân loại
    label_encoders = {}
    categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    st.write("### Dữ liệu sau khi mã hóa các biến phân loại:")
    st.dataframe(data)

    # Bước 4: Loại bỏ cột không cần thiết và chuẩn hóa
    if 'Name' in data.columns:
        data.drop(columns=['Name'], inplace=True)

    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    st.write("### Dữ liệu sau chuẩn hóa:")
    st.dataframe(data)

    # Bước 5: Chia tập dữ liệu
    test_size = 0.15
    train_size = st.slider("Chọn phần trăm dữ liệu train", 50, 85, 70) / 100
    valid_size = 1 - train_size

    train, test = train_test_split(data, test_size=test_size, random_state=42)
    train, valid = train_test_split(train, test_size=valid_size / (train_size + valid_size), random_state=42)

    X_train, y_train = train.drop(columns=['Survived']), train['Survived']
    X_valid, y_valid = valid.drop(columns=['Survived']), valid['Survived']
    X_test, y_test = test.drop(columns=['Survived']), test['Survived']

    # Bước 6: Huấn luyện mô hình với MLflow
    exp = mlflow.set_experiment("Titanic_Model_Training")
    model_choice = st.selectbox("Chọn mô hình", ["Multiple Regression", "Polynomial Regression", "Random Forest"])
    cv_folds = st.slider("Chọn số lượng folds cho Cross Validation", 2, 20, 10)

    if model_choice == "Multiple Regression":
        model = Ridge(alpha=1.0)
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(model, param_grid, cv=cv_folds)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        with mlflow.start_run(experiment_id=exp.experiment_id):
            y_pred = best_model.predict(X_valid)
            mse = mean_squared_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)
            mlflow.log_metric("MSE_Multiple_Regression", mse)
            mlflow.log_metric("R2_Multiple_Regression", r2)
            mlflow.sklearn.log_model(best_model, "multiple_regression", input_example=X_valid.iloc[:1])

    elif model_choice == "Polynomial Regression":
        degree = st.slider("Chọn bậc của đa thức", 2, 5, 2)
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        param_grid = {'ridge__alpha': [0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(model, param_grid, cv=cv_folds)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        with mlflow.start_run(experiment_id=exp.experiment_id):
            y_pred = best_model.predict(X_valid)  # Pipeline tự động xử lý biến đổi
            mse = mean_squared_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)
            mlflow.log_metric("MSE_Polynomial_Regression", mse)
            mlflow.log_metric("R2_Polynomial_Regression", r2)
            mlflow.sklearn.log_model(best_model, "polynomial_regression", input_example=X_valid.iloc[:1])

    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        with mlflow.start_run(experiment_id=exp.experiment_id):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            mse = mean_squared_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)
            mlflow.log_metric("MSE_Random_Forest", mse)
            mlflow.log_metric("R2_Random_Forest", r2)
            mlflow.sklearn.log_model(model, "random_forest", input_example=X_valid.iloc[:1])

    st.write(f"MSE của mô hình {model_choice}: {mse}")
    st.write(f"R2 Score của mô hình {model_choice}: {r2}")
if __name__ == "__main__":
    main()