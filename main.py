import streamlit as st
import os
import importlib.util

# 🏆 Tiêu đề trang web
st.title("Bộ sưu tập bài tập 🎯")

# # 🗂 Lấy danh sách các bài tập từ thư mục "exercises"
# exercise_files = [f for f in os.listdir("exercises") if f.endswith(".py")]
# exercise_names = [f.replace(".py", "") for f in exercise_files]

# Kiểm tra thư mục exercises có tồn tại không
if not os.path.exists("exercises"):
    st.error("Thư mục 'exercises' không tồn tại! Vui lòng kiểm tra lại.")
else:
    exercise_files = [f for f in os.listdir("exercises") if f.endswith(".py")]
    exercise_names = [f.replace(".py", "") for f in exercise_files]

# 🎛 Tạo menu sidebar để chọn bài tập
selected_exercise = st.sidebar.selectbox("Chọn bài tập", exercise_names)

# 📌 Load và chạy bài tập khi chọn
if selected_exercise:
    file_path = f"exercises/{selected_exercise}.py"

    # Load module bài tập
    spec = importlib.util.spec_from_file_location(selected_exercise, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Gọi hàm `main()` của bài tập
    if hasattr(module, "main"):
        module.main()
    else:
        st.error(f"Bài tập {selected_exercise} không có hàm main()!")
