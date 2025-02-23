import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Streamlit UI
st.title("Dự đoán Titanic Survival")

def main():
    # Bước 1: Đọc và đánh giá dữ liệu
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu (CSV hoặc TXT)", type=["csv", "txt"])
    if uploaded_file is None:
        st.warning("Vui lòng tải lên tệp dữ liệu để tiếp tục.")
        return
    
    data = pd.read_csv(uploaded_file)
    st.write("### Dữ liệu ban đầu:")
    st.dataframe(data)

    # Bước 1.2: Kiểm tra giá trị bị thiếu
    missing_values = data.isnull().sum()
    st.write("### Giá trị bị thiếu:")
    st.write(missing_values[missing_values > 0])
 
    # Bước 2: Điền giá trị bị thiếu
    columns_to_fill = st.multiselect("Chọn các cột cần điền giá trị bị thiếu", data.columns, default=[col for col in ["Age", "Cabin", "Embarked"] if col in data.columns])
    
    fill_methods = {}
    for column in columns_to_fill:
        if data[column].dtype == 'object':
            fill_methods[column] = st.selectbox(f"Chọn phương pháp điền cho {column}", ["Mode"])
        else:
            fill_methods[column] = st.selectbox(f"Chọn phương pháp điền cho {column}", ["Trung vị", "Trung bình", "Mode"])
    
    for column, method in fill_methods.items():
        if method == "Mode":
            data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == "Trung vị":
            data[column].fillna(data[column].median(), inplace=True)
        elif method == "Trung bình":
            data[column].fillna(data[column].mean(), inplace=True)
    
    # Chuyển đổi cột Age thành số nguyên trong khoảng 1-100
    if "Age" in data.columns:
        data['Age'] = data['Age'].apply(lambda x: int(round(x)) if pd.notnull(x) else x)
        data['Age'] = np.clip(data['Age'], 1, 100)
    
    # Chuẩn hóa cột Cabin chỉ chứa ký tự và số hợp lệ
    if "Cabin" in data.columns:
        data['Cabin'] = data['Cabin'].astype(str).str.extract(r'([A-Za-z]+\d*)')[0]
    
    st.write("### Dữ liệu sau khi điền giá trị thiếu và xử lý cột Age, Cabin:")
    st.dataframe(data)
    
    # Bước 3: Chuyển đổi dữ liệu phân loại
    encoding_method = st.selectbox("Chọn phương pháp mã hóa", ["Label Encoding", "One-Hot Encoding"])
    categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket']
    if encoding_method == "Label Encoding":
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    else:
        data = pd.get_dummies(data, columns=categorical_columns)

    st.write("### Dữ liệu sau khi mã hóa:")
    st.dataframe(data)
    
        # Bước 4: Loại bỏ cột không cần thiết và chuẩn hóa
    columns_to_drop = st.multiselect("Chọn các cột cần loại bỏ", data.columns, default=[])
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True)
    
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if not ((data[col].min() >= 0) and (data[col].max() <= 1)):
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    
    st.write("### Dữ liệu sau chuẩn hóa:")
    st.dataframe(data.head())
    
   # Bước 5: Chia tập dữ liệu
    test_size = st.slider("Chọn phần trăm dữ liệu test", 10, 30, 15) / 100
    train_size = st.slider("Chọn phần trăm dữ liệu train", 50, 90 - int(test_size * 100), 70) / 100
    valid_size = 1 - train_size - test_size

    train, test = train_test_split(data, test_size=test_size, random_state=42)
    train, valid = train_test_split(train, test_size=valid_size / (train_size + valid_size), random_state=42)

    X_train, y_train = train.drop(columns=['Survived']), train['Survived']
    X_valid, y_valid = valid.drop(columns=['Survived']), valid['Survived']
    X_test, y_test = test.drop(columns=['Survived']), test['Survived']
    
    st.write("### Kích thước tập dữ liệu sau khi chia:")
    st.write(f"Train size: {X_train.shape}, Validation size: {X_valid.shape}, Test size: {X_test.shape}")
    
    # Bước 6: Huấn luyện mô hình
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    st.write("### Mean Squared Error trên tập Validation:", mse)
    
    if mse > 0.2:
        st.warning("Cần cải thiện model để giảm lỗi MSE dưới 0.2")
    
if __name__ == "__main__":
    main()
