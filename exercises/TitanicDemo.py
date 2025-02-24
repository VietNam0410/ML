import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def preprocess_data(df):
    # Xử lý giá trị thiếu
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    
    # Chuyển đổi dữ liệu phân loại thành số
    categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket']
    label_encoders = {}
    category_values = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        category_values[col] = le.classes_  # Lưu danh sách giá trị có sẵn
    
    # Loại bỏ cột 'Name' nếu có
    if 'Name' in df.columns:
        df.drop(columns=['Name'], inplace=True)

    return df, label_encoders, category_values

def train_model(X, y, model_type="Random Forest"):
    # Tách tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Chọn và huấn luyện mô hình dựa trên lựa chọn
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # Logistic Regression
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title("Dự đoán Sống Sót Trên Tàu Titanic")
    
    # Nút để tải và xử lý dữ liệu
    if st.button("Tải và Xử Lý Dữ Liệu"):
        with st.spinner("Đang tải và xử lý dữ liệu..."):
            df = load_data()
            df, label_encoders, category_values = preprocess_data(df)
        
        # Lưu dữ liệu đã xử lý vào session_state để tái sử dụng
        st.session_state.df = df
        st.session_state.label_encoders = label_encoders
        st.session_state.category_values = category_values
        
        st.success("Dữ liệu đã được tải và xử lý thành công!")
        st.write("### Dữ liệu đã xử lý:")
        st.dataframe(df.head())
    
    # Nếu dữ liệu đã được xử lý, tiếp tục với dự đoán
    if 'df' in st.session_state:
        df = st.session_state.df
        label_encoders = st.session_state.label_encoders
        category_values = st.session_state.category_values
        
        # Tách tập dữ liệu
        X = df.drop(columns=['Survived'])
        y = df['Survived']
        
        # Chọn mô hình
        model_choice = st.selectbox("Chọn mô hình để dự đoán", ["Random Forest", "Logistic Regression"])
        
        # Huấn luyện mô hình
        model, X_test, y_test = train_model(X, y, model_choice)
        
        st.header("1. Chọn Dữ Liệu Để Dự Đoán")
        input_data = {}
        for col in X.columns:
            if col in category_values:  # Cột phân loại
                input_data[col] = st.selectbox(f"Chọn giá trị cho {col}", category_values[col])
            else:  # Cột số
                min_val, max_val = df[col].min(), df[col].max()
                input_data[col] = st.number_input(f"Nhập giá trị cho {col}", float(min_val), float(max_val))
        
        # Chuyển đổi dữ liệu đầu vào
        input_df = pd.DataFrame([input_data])
        
        # Chuyển đổi dữ liệu phân loại thành số
        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        st.header("2. Kết Quả Dự Đoán")
        if st.button("Dự đoán sống sót"):
            prediction = model.predict(input_df)[0]
            st.subheader(f"Kết quả: {'Sống sót' if prediction == 1 else 'Không sống sót'}")

            # Kiểm tra dữ liệu có trong tập Titanic không
            match = df[(X == input_df.iloc[0]).all(axis=1)]
            if not match.empty:
                true_value = match["Survived"].values[0]
                is_correct = prediction == true_value
                st.write(f"🔍 **So sánh với dữ liệu gốc:** {'Đúng' if is_correct else 'Sai'}")
            else:
                st.write("🆕 **Dữ liệu này không có trong tập Titanic gốc!**")
        
        # Đánh giá mô hình (tùy chọn)
        st.header("3. Đánh giá mô hình")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Độ chính xác trên tập test: {accuracy:.4f}")

if __name__ == "__main__":
    main()