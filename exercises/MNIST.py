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
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load dữ liệu (Caching to improve performance)
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

# Cache the scaler object (use st.cache_resource for objects like scalers)
@st.cache_resource
def get_scaler():
    return StandardScaler()

# Preprocess dữ liệu
def preprocess_data(X_train, X_test, scaler):
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def main():
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = get_scaler()

    # Tạo các tab
    tab1, tab2 = st.tabs(["🔧 Xử lý dữ liệu & Huấn luyện mô hình", "🎨 Demo mô hình"])

    with tab1:
        st.header("1. Xử lý dữ liệu và Huấn luyện mô hình")
        
        # Load dữ liệu
        X, y = load_mnist()
        st.write(f"🔹 Dữ liệu MNIST có {X.shape[0]} hình ảnh, mỗi ảnh có {X.shape[1]} pixel")

        # Chia dữ liệu
        test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.success(f"✅ Dữ liệu đã được chia thành công với tỷ lệ kiểm tra là {test_size:.2f}!")

        # Preprocess dữ liệu
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test, st.session_state.scaler)

        # Chọn mô hình
        model_option = st.selectbox("Chọn mô hình để huấn luyện", ["Decision Tree", "SVM"])

        # Tham số mô hình
        if model_option == "Decision Tree":
            max_depth = st.slider("Max Depth (Decision Tree)", 1, 20, 10)
        else:
            C_value = st.slider("C (SVM)", 0.1, 10.0, 1.0)

        # Huấn luyện mô hình
        if st.button("Huấn luyện mô hình"):
            st.write("⏳ Đang huấn luyện mô hình...")
            if model_option == "Decision Tree":
                st.session_state.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            else:
                st.session_state.model = SVC(kernel='rbf', C=C_value)
            
            st.session_state.model.fit(X_train_scaled, y_train)
            y_pred = st.session_state.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"✅ Độ chính xác: {accuracy:.4f}")
            
            # Hiển thị ma trận nhầm lẫn
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.matshow(confusion_matrix(y_test, y_pred), cmap=plt.cm.Blues)
            st.pyplot(fig)

            # Lưu vào MLflow (with error handling)
            try:
                with mlflow.start_run():
                    mlflow.sklearn.log_model(st.session_state.model, "model")
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_param("model", model_option)
                    mlflow.log_param("test_size", test_size)
                st.success("✅ Mô hình đã được lưu trữ trong MLflow!")
            except Exception as e:
                st.warning(f"⚠️ Lỗi khi lưu vào MLflow: {e}")

            st.info("🎉 Mô hình huấn luyện thành công! Chuyển sang tab 2 để demo.")

    with tab2:
        st.header("2. Demo mô hình")

        if st.session_state.model is None:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi thử demo!")
            st.info("🎯 Chuyển sang tab 1 để huấn luyện mô hình.")
        else:
            st.subheader("Vẽ số hoặc tải ảnh để mô hình dự đoán")
            
            # Tải ảnh hoặc vẽ
            uploaded_file = st.file_uploader("Tải ảnh chữ số (28x28 pixel)", type=['png', 'jpg', 'jpeg'])
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

            def preprocess_image(image, scaler):
                image = image.convert('L')  # Convert to grayscale
                image = ImageOps.invert(image)  # Invert colors
                image = image.resize((28, 28))  # Resize to 28x28
                image_array = np.array(image).reshape(1, -1)  # Flatten
                image_scaled = scaler.transform(image_array)  # Apply the same scaling
                return image_scaled

            # Dự đoán
            if uploaded_file or (canvas_result.image_data is not None):
                if uploaded_file:
                    image = Image.open(uploaded_file)
                else:
                    image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))

                image_scaled = preprocess_image(image, st.session_state.scaler)
                prediction = st.session_state.model.predict(image_scaled)[0]
                st.image(image, caption=f"📢 Mô hình dự đoán: {prediction}", use_container_width=True)
                st.success(f"✅ Kết quả dự đoán: {prediction}")
                st.info(f"🎉 Dự đoán thành công! Chữ số bạn vẽ là: {prediction}")
            else:
                st.info("✏️ Vẽ số hoặc tải ảnh để thử dự đoán.")

if __name__ == "__main__":
    main()