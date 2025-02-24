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
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Styles cho UI
def set_custom_styles():
    st.markdown("""
        <style>
        .main > div {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin: 1rem 0;
        }
        .upload-text {
            text-align: center;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Load dữ liệu với progress bar
@st.cache_data(show_spinner=False)
def load_mnist():
    with st.spinner('Đang tải dữ liệu MNIST...'):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        return X, y

# Chuẩn hóa dữ liệu với feedback
@st.cache_data(show_spinner=False)
def preprocess_data(X_train, X_test):
    with st.spinner('Đang chuẩn hóa dữ liệu...'):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

def display_model_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(
        label="Độ chính xác",
        value=f"{accuracy:.2%}",
        delta=f"{(accuracy-0.5)*100:.1f}% so với baseline"
    )

def process_drawing(image_data, model, scaler):
    # Xử lý ảnh vẽ
    image = Image.fromarray((image_data[:, :, :3] * 255).astype(np.uint8))
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    # Hiển thị ảnh đã xử lý
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc", width=200)
    
    # Chuẩn hóa và dự đoán
    image_array = np.array(image).reshape(1, -1)
    image_scaled = scaler.transform(image_array)
    prediction = model.predict(image_scaled)[0]
    
    with col2:
        st.success(f"Kết quả dự đoán: {prediction}")
        confidence = model.predict_proba(image_scaled)[0] if hasattr(model, 'predict_proba') else None
        if confidence is not None:
            st.progress(float(max(confidence)))
            st.text(f"Độ tin cậy: {max(confidence):.2%}")

def train_model_tab():
    st.header("Huấn luyện mô hình")
    
    # Tải và chia dữ liệu với thanh tiến trình
    with st.expander("📊 Cấu hình dữ liệu", expanded=True):
        X, y = load_mnist()
        total_samples = X.shape[0]
        
        test_size = st.slider(
            "Tỷ lệ dữ liệu kiểm tra",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Chọn tỷ lệ dữ liệu dùng để kiểm tra mô hình"
        )
        
        train_samples = int(total_samples * (1 - test_size))
        test_samples = int(total_samples * test_size)
        
        st.info(f"""📌 Phân chia dữ liệu:
        - Tổng số mẫu: {total_samples:,}
        - Số mẫu huấn luyện: {train_samples:,} ({(1-test_size):.0%})
        - Số mẫu kiểm tra: {test_samples:,} ({test_size:.0%})""")

    # Cấu hình mô hình
    with st.expander("⚙️ Cấu hình mô hình", expanded=True):
        model_option = st.selectbox(
            "Chọn loại mô hình",
            ["Decision Tree", "SVM"],
            help="Chọn thuật toán học máy để phân loại"
        )

        if model_option == "Decision Tree":
            max_depth = st.slider(
                "Độ sâu tối đa của cây",
                min_value=1,
                max_value=20,
                value=10,
                help="Kiểm soát độ phức tạp của mô hình"
            )
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            C_value = st.slider(
                "Hệ số điều chỉnh C",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Kiểm soát mức độ phạt với các điểm phân loại sai"
            )
            model = SVC(kernel='rbf', C=C_value, probability=True)

    # Huấn luyện mô hình
    if st.button("🚀 Bắt đầu huấn luyện", help="Nhấn để bắt đầu quá trình huấn luyện"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        
        with st.spinner('Đang huấn luyện mô hình...'):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Lưu mô hình và scaler vào session state
            st.session_state.model = model
            st.session_state.scaler = StandardScaler().fit(X_train)
            
            # Hiển thị metrics
            display_model_metrics(y_test, y_pred)
            
            # Log với MLFlow
            with mlflow.start_run():
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                mlflow.log_param("model_type", model_option)
                mlflow.log_param("test_size", test_size)

def test_model_tab():
    st.header("Thử nghiệm mô hình")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi thử nghiệm!")
        return
    
    method = st.radio(
        "Chọn phương thức nhập liệu",
        ["✏️ Vẽ", "📁 Tải ảnh"],
        horizontal=True
    )
    
    if method == "✏️ Vẽ":
        st.markdown("### Vẽ số cần nhận dạng")
        canvas_result = st_canvas(
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if canvas_result.image_data is not None:
            process_drawing(
                canvas_result.image_data,
                st.session_state.model,
                st.session_state.scaler
            )
    
    else:
        uploaded_file = st.file_uploader(
            "Tải lên ảnh chữ số",
            type=['png', 'jpg', 'jpeg'],
            help="Chọn ảnh chữ số cần nhận dạng (nền trắng, chữ đen)"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            process_drawing(
                np.array(image),
                st.session_state.model,
                st.session_state.scaler
            )

def main():
    # Đặt styles
    set_custom_styles()
    
    st.title("🔢 Phân loại chữ số viết tay MNIST")
    st.markdown("""
    ### 👋 Chào mừng bạn đến với ứng dụng phân loại chữ số MNIST!
    Ứng dụng này cho phép bạn:
    - Thử nghiệm với các mô hình khác nhau
    - Điều chỉnh tham số để tối ưu hiệu suất
    - Vẽ hoặc tải lên chữ số để kiểm tra
    """)

    # Tạo tabs
    tab1, tab2 = st.tabs(["🔧 Huấn luyện mô hình", "🎨 Thử nghiệm"])
    
    with tab1:
        train_model_tab()
    
    with tab2:
        test_model_tab()

if __name__ == "__main__":
    main()