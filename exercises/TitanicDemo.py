import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# Nhớ tạo thêm chọn model để dự đoán
def load_data():
    df = pd.read_csv('titanic.csv')#Nếu dùng web app thì thay đường dẫn bằng 'titanic.csv'
   #còn local thì thay bằng '/Users/nguyenvietnam/Documents/Machine_Learning/titanic.csv'
    return df

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

def main():
    st.title("Dự đoán sống sót trên tàu Titanic")
    
    df = load_data()
    df, label_encoders, category_values = preprocess_data(df)
    
    # Tách tập dữ liệu
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Huấn luyện mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
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

        # ✅ **Kiểm tra dữ liệu có trong tập Titanic không**
        match = df[(X == input_df.iloc[0]).all(axis=1)]
        if not match.empty:
            true_value = match["Survived"].values[0]
            is_correct = prediction == true_value
            st.write(f"🔍 **So sánh với dữ liệu gốc:** {is_correct}")
        else:
            st.write("🆕 **Dữ liệu này không có trong tập Titanic gốc!**")
    
if __name__ == "__main__":
    main()
