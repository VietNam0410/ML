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
# + Nhớ thêm tính năng tải file (csv hoặc txt) lên 
# Streamlit UI
st.title("Dự đoán Titanic Survival")
def main():
    # # Bước 1: Đọc và đánh giá dữ liệu
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu (CSV hoặc TXT)", type=["csv", "txt"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("titanic.csv")
    st.write("### Dữ liệu ban đầu:")
    st.dataframe(data)

    # Bước 1.2: Kiểm tra giá trị bị thiếu
    missing_values = data.isnull().sum()
    st.write("### Giá trị bị thiếu:")
    st.write(missing_values[missing_values > 0])
 
    # Bước 2: Điền giá trị bị thiếu
    fill_method = st.selectbox("Chọn phương pháp điền giá trị thiếu", ["Trung vị", "Trung bình", "Mode"])  # ĐÃ THÊM
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            if fill_method == "Trung vị":
                data[column].fillna(data[column].median(), inplace=True)
            elif fill_method == "Trung bình":
                data[column].fillna(data[column].mean(), inplace=True)

    st.write("### Dữ liệu sau khi điền giá trị thiếu:")
    st.dataframe(data)
    # Bước 3: Chuyển đổi dữ liệu phân loại
    # + Thêm tính năng chọn các kiểu chuyển đổi
    encoding_method = st.selectbox("Chọn phương pháp mã hóa", ["Label Encoding", "One-Hot Encoding"])  # ĐÃ THÊM
    label_encoders = {}
    categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket']
    if encoding_method == "Label Encoding":
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    else:
        data = pd.get_dummies(data, columns=categorical_columns)

    st.write("### Dữ liệu sau khi mã hóa các biến phân loại:")
    st.dataframe(data)
    # Bước 4: Loại bỏ cột không cần thiết và chuẩn hoá tất cả các cột
    # + Thêm tính năng xoá các cột tự chọn theo người dùng
    columns_to_drop = st.multiselect("Chọn các cột cần loại bỏ", data.columns, default=[])  # ĐÃ SỬA ĐỂ ĐẢM BẢO NAME CŨNG ĐƯỢC CHUẨN HÓA
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True)
    # Chuẩn hóa dữ liệu (bao gồm cả cột Name nếu không bị loại bỏ)
    # Nhớ đưa về miền giá trị hợp lý vd:Tuổi 0-100, sex: 0 - 1,..
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    st.write("### Dữ liệu sau chuẩn hóa:")
    st.dataframe(data.head())

    # Bước 5: Chia tập dữ liệu
    # +Thêm tính năng chọn tỉ lệ của test
    def update_splits():
        global test_size, train_size, valid_size
        test_size = st.slider("Chọn phần trăm dữ liệu test", 10, 30, 15) / 100  # ĐÃ THÊM
        train_size = st.slider("Chọn phần trăm dữ liệu train", 50, 90 - int(test_size * 100), 70) / 100  # ĐÃ SỬA
        valid_size = 1 - train_size - test_size
    update_splits()

    train, test = train_test_split(data, test_size=test_size, random_state=42)
    train, valid = train_test_split(train, test_size=valid_size / (train_size + valid_size), random_state=42)

    X_train, y_train = train.drop(columns=['Survived']), train['Survived']
    X_valid, y_valid = valid.drop(columns=['Survived']), valid['Survived']
    X_test, y_test = test.drop(columns=['Survived']), test['Survived']
    
    st.write("### Kích thước tập dữ liệu sau khi chia:")
    st.write(f"Train size: {X_train.shape}, Validation size: {X_valid.shape}, Test size: {X_test.shape}")
    
    st.write("### Mẫu dữ liệu tập Test:")  # ĐÃ THÊM
    st.dataframe(X_test.head())  # ĐÃ THÊM
    # Bước 6: Huấn luyện mô hình với MLflow
    exp = mlflow.set_experiment("Titanic_Model_Training")
    st.write("### Giới thiệu về Cross Validation")
    st.write("Cross Validation là kỹ thuật giúp đánh giá mô hình bằng cách chia dữ liệu thành nhiều phần nhỏ (folds), huấn luyện và kiểm tra trên từng phần, sau đó lấy trung bình kết quả để có đánh giá tổng quát nhất. Điều này giúp giảm tình trạng overfitting và cải thiện độ tin cậy của mô hình.")

    model_choice = st.selectbox("Chọn mô hình", ["Multiple Regression", "Polynomial Regression", "Random Forest"])
    # Giới Thiệu Cross Validation
   
    # cv_folds = st.slider("Chọn số lượng folds cho Cross Validation", 2, 20, 10)
    # model_choice = st.selectbox("Chọn mô hình", ["Multiple Regression", "Polynomial Regression", "Random Forest"])
    cv_folds = st.slider("Chọn số lượng folds cho Cross Validation", 2, 20, 10)  # ĐÃ THÊM

    #Mutltiple Regression
    # Hiện các tham số của mô hình ra
    
    if model_choice == "Multiple Regression":
        alpha = st.slider("Chọn giá trị alpha", 0.01, 10.0, 1.0)  # ĐÃ THÊM
        model = Ridge(alpha=alpha)
        best_model = model
        best_model.fit(X_train, y_train)
        
    elif model_choice == "Polynomial Regression":
        degree = st.slider("Chọn bậc của đa thức", 2, 5, 2)  # ĐÃ THÊM
        alpha = st.slider("Chọn giá trị alpha cho Ridge", 0.01, 10.0, 1.0)  # ĐÃ THÊM
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        best_model = model
        best_model.fit(X_train, y_train)
        
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Chọn số lượng cây trong Random Forest", 10, 200, 100)  # ĐÃ THÊM
        max_depth = st.slider("Chọn độ sâu tối đa của cây", 2, 20, 10)  # ĐÃ THÊM
        best_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        best_model.fit(X_train, y_train)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        y_pred = best_model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)
        mlflow.log_metric(f"MSE_{model_choice}", mse)
        mlflow.log_metric(f"R2_{model_choice}", r2)
        mlflow.sklearn.log_model(best_model, model_choice.lower().replace(" ", "_"), input_example=X_valid.iloc[:1])
    # + Train model xong nhớ tạo thêm nútlog vào mlflow
    st.write(f"MSE của mô hình {model_choice}: {mse}")
    st.write(f"R2 Score của mô hình {model_choice}: {r2}")
    # Nút log vào MLflow sau khi train model
    if st.button("Log Model vào MLflow"):  # ĐÃ THÊM
        with mlflow.start_run(experiment_id=exp.experiment_id):
            mlflow.sklearn.log_model(best_model, f"{model_choice}_final")
        st.write("Mô hình đã được log vào MLflow thành công!")  # ĐÃ THÊM
if __name__ == "__main__":
    main()