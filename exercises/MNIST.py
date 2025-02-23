import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image, ImageOps
# import cv2
from streamlit_drawable_canvas import st_canvas

# Load dữ liệu
@st.cache_data
def load_mnist():
    mnist = openml.datasets.get_dataset(45104)  # Dataset MNIST-784 trên OpenML
    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)  # Lấy dữ liệu
    X = pd.DataFrame(X)  # Chuyển X thành DataFrame
    y = y.astype(int)  # Đảm bảo y là kiểu int
    return X, y

def bai_tap_mnist():
    st.subheader("📝 Bài tập MNIST")
    st.write("Hãy thử thay đổi các tham số của mô hình và xem ảnh hưởng của chúng đến độ chính xác.")

def main():
    # Title
    st.title("🎨 Nhận diện chữ số MNIST với SVM & Decision Tree")
    st.markdown("""
    ### 📌 Hướng dẫn:
    1. **Tải dữ liệu MNIST** 📥
    2. **Chọn mô hình và huấn luyện** 🤖
    3. **Xem đánh giá kết quả** 🎯
    4. **Tải ảnh vẽ tay hoặc vẽ trực tiếp để dự đoán** 🖌
    """)
    
    X, y = load_mnist()
    st.write(f"🔹 Dữ liệu MNIST có {X.shape[0]} hình ảnh, mỗi ảnh có {X.shape[1]} pixel")

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Chọn mô hình
    st.sidebar.header("⚙️ Cài đặt mô hình")
    model_option = st.sidebar.selectbox("Chọn mô hình để huấn luyện", ["Decision Tree", "SVM"])

    if st.sidebar.button("Huấn luyện mô hình"):
        st.sidebar.write("⏳ Đang huấn luyện...")
        if model_option == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
        else:
            model = SVC(kernel='rbf', C=10)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.sidebar.success(f"✅ Độ chính xác: {accuracy:.4f}")
        
        # Hiển thị ma trận nhầm lẫn
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.matshow(confusion_matrix(y_test, y_pred), cmap=plt.cm.Blues)
        st.pyplot(fig)
        
        # Lưu vào MLFlow
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_param("model", model_option)

    # Tải ảnh vẽ để dự đoán
    st.subheader("🖌 Vẽ số và dự đoán")
    uploaded_file = st.file_uploader("Tải ảnh chữ số (28x28 pixel)", type=['png', 'jpg', 'jpeg'])

    # Thêm bảng vẽ cho người dùng
    st.subheader("✏️ Vẽ số trực tiếp")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    def preprocess_image(image):
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image_array = np.array(image).reshape(1, -1)
        return scaler.transform(image_array)

    # Kiểm tra nếu có dữ liệu từ bảng vẽ hoặc file upload
    if uploaded_file or (canvas_result is not None and hasattr(canvas_result, 'image_data') and canvas_result.image_data is not None):
        if uploaded_file:
            image = Image.open(uploaded_file)
        elif canvas_result is not None and hasattr(canvas_result, 'image_data') and canvas_result.image_data is not None:
            image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))

        # Kiểm tra nếu ảnh hợp lệ
        if image is not None:
            image_array = preprocess_image(image)
            
            if model_option == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train_scaled, y_train)
            else:
                model = SVC(kernel='rbf', C=10).fit(X_train_scaled, y_train)
            
            prediction = model.predict(image_array)[0]
            st.image(image, caption=f"📢 Mô hình dự đoán: {prediction}", use_column_width=True)
            st.success(f"✅ Kết quả dự đoán: {prediction}")
        else:
            st.error("⚠️ Vui lòng vẽ số hoặc tải ảnh hợp lệ.")

    bai_tap_mnist()

if __name__ == "__main__":
    main()
