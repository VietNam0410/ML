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

# Caching chuẩn hóa dữ liệu
@st.cache_data
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def bai_tap_mnist():
    st.subheader("📝 Bài tập MNIST")
    st.write("Hãy thử thay đổi các tham số của mô hình và xem ảnh hưởng của chúng đến độ chính xác.")

def main():
    # Tạo các tab
    tab1, tab2 = st.tabs(["🔧 Xử lý dữ liệu & Huấn luyện mô hình", "🎨 Demo mô hình"])

    with tab1:
        st.header("1. Xử lý dữ liệu và Huấn luyện mô hình")
        
        # Load dữ liệu
        X, y = load_mnist()
        st.write(f"🔹 Dữ liệu MNIST có {X.shape[0]} hình ảnh, mỗi ảnh có {X.shape[1]} pixel")

        # Chia dữ liệu (Cho phép người dùng điều chỉnh tỷ lệ chia dữ liệu)
        test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Thông báo sau khi chia dữ liệu
        st.success(f"✅ Dữ liệu đã được chia thành công với tỷ lệ kiểm tra là {test_size:.2f}!")

        # Preprocess dữ liệu (caching)
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

        # Chọn mô hình
        model_option = st.selectbox("Chọn mô hình để huấn luyện", ["Decision Tree", "SVM"])

        # Thêm tham số mô hình có thể điều chỉnh
        if model_option == "Decision Tree":
            max_depth = st.slider("Max Depth (Decision Tree)", 1, 20, 10)
        else:
            C_value = st.slider("C (SVM)", 0.1, 10.0, 1.0)

        # Lưu mô hình vào session_state để tránh huấn luyện lại
        if 'model' not in st.session_state:
            st.session_state.model = None

        # Huấn luyện mô hình nếu chưa có mô hình đã huấn luyện
        if st.button("Huấn luyện mô hình") and st.session_state.model is None:
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

            # Lưu vào MLFlow với input_example
            input_example = X_train_scaled[0].reshape(1, -1)  # Ví dụ input từ X_train
            with mlflow.start_run():
                mlflow.sklearn.log_model(st.session_state.model, "model", input_example=input_example)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("model", model_option)
                mlflow.log_param("test_size", test_size)

            st.success("✅ Mô hình đã được huấn luyện và lưu trữ!")

            # Thông báo thành công
            st.info("🎉 Mô hình huấn luyện thành công! Bây giờ bạn có thể chuyển sang tab 2 để demo mô hình.")

    with tab2:
        st.header("2. Demo mô hình")

        if st.session_state.model is not None:
            st.subheader("Vẽ số hoặc tải ảnh để mô hình dự đoán")
            
            # Tải ảnh vẽ hoặc file
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
                return image_array

            # Kiểm tra nếu có dữ liệu từ bảng vẽ hoặc file upload
            if uploaded_file or (canvas_result.image_data is not None):
                if uploaded_file:
                    image = Image.open(uploaded_file)
                elif canvas_result.image_data is not None:
                    image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))

                # Kiểm tra nếu ảnh hợp lệ
                if image is not None:
                    image_array = preprocess_image(image)
                    
                    # Dự đoán với mô hình đã huấn luyện
                    prediction = st.session_state.model.predict(image_array)[0]
                    st.image(image, caption=f"📢 Mô hình dự đoán: {prediction}", use_container_width=True)
                    st.success(f"✅ Kết quả dự đoán: {prediction}")

                    # Thông báo sau khi dự đoán thành công
                    st.info(f"🎉 Dự đoán thành công! Chữ số bạn vẽ là: {prediction}")

                else:
                    st.error("⚠️ Vui lòng vẽ số hoặc tải ảnh hợp lệ.")
        else:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi thử demo!")
            st.info("🎯 Chuyển sang tab 1 để huấn luyện mô hình trước khi thử demo.")

if __name__ == "__main__":
    main()
