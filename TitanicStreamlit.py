import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

def main():
    # Đọc tập dữ liệu Titanic
    df = pd.read_csv('/Users/nguyenvietnam/Documents/Machine_Learning/titanic.csv')

    st.title("Hướng Dẫn Tiền Xử Lý Dữ Liệu Titanic")

    st.header("1. Xem Dữ Liệu Ban Đầu")
    st.write("Dữ liệu gốc trước khi tiền xử lý:")
    st.dataframe(df)

    st.header("2. Kiểm Tra Giá Trị Thiếu")
    st.write("Thông tin dữ liệu và giá trị bị thiếu:")
    st.text(df.info())
    st.text(df.isnull().sum())

    # Xử lý dữ liệu: điền giá trị thiếu tùy theo từng cột
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)

    st.header("3. Điền Giá Trị Thiếu")
    st.write("Dữ liệu sau khi điền giá trị thiếu:")
    st.dataframe(df)

    # Chuyển đổi dữ liệu phân loại thành số nguyên tuần tự
    label_encoders = {}
    categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    st.header("4. Chuyển Đổi Dữ Liệu Phân Loại")
    st.write("Dữ liệu sau khi mã hóa các biến phân loại thành số:")
    st.dataframe(df)

    # Chuẩn hóa các cột số (loại bỏ cột 'Name' nếu tồn tại)
    if 'Name' in df.columns:
        df.drop(columns=['Name'], inplace=True)

    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    st.header("5. Chuẩn Hóa Dữ Liệu")
    st.write("Dữ liệu sau khi chuẩn hóa:")
    st.dataframe(df)

    # Người dùng chọn tỷ lệ train/validation, giữ nguyên tập test 15%
    train_size = st.slider("Chọn tỷ lệ tập train (%)", 50, 85, 70) / 100
    valid_size = 1 - train_size

    temp_data, test_data = train_test_split(df, test_size=0.15, random_state=42)
    train_data, valid_data = train_test_split(temp_data, test_size=valid_size/(train_size + valid_size), random_state=42)

    st.header("6. Chia Dữ Liệu")
    st.write(f'Train size: {train_data.shape}, Validation size: {valid_data.shape}, Test size: {test_data.shape}')
    st.write("Dữ liệu sau khi chia:")
    st.dataframe(train_data)

    # Lựa chọn mô hình
    st.header("7. Huấn Luyện Mô Hình")
    model_choice = st.selectbox("Chọn mô hình học máy", ["Multiple Regression", "Polynomial Regression", "Random Forest"])
    cv_folds = st.slider("Chọn số lượng folds cho Cross Validation", 2, 20, 10)

    X_train = train_data.drop(columns=['Survived'])
    y_train = train_data['Survived']
    X_valid = valid_data.drop(columns=['Survived'])
    y_valid = valid_data['Survived']

    if model_choice == "Multiple Regression":
        model = Ridge(alpha=1.0)
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(model, param_grid, cv=cv_folds)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    elif model_choice == "Polynomial Regression":
        degree = st.slider("Chọn bậc của đa thức", 2, 5, 2)
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        param_grid = {'ridge__alpha': [0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(model, param_grid, cv=cv_folds)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

    # Đánh giá mô hình
    train_score = model.score(X_train, y_train)
    valid_score = model.score(X_valid, y_valid)

    st.write(f"Độ chính xác trên tập huấn luyện: {train_score:.4f}")
    st.write(f"Độ chính xác trên tập validation: {valid_score:.4f}")

    # # Tạo giao diện cho người dùng nhập dữ liệu để dự đoán
    # st.header("8. Dự Đoán Trực Tiếp")
 

if __name__ == "__main__":
    main()
