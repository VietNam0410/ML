import numpy as np
from sklearn.datasets import fetch_openml

# Load dữ liệu MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# Chuyển đổi kiểu dữ liệu
X = X.astype(np.float32)
y = y.astype(np.int64)

# Tính số hàng tối đa cho mỗi file (< 25MB)
bytes_per_row = 784 * 4  # 3136 byte mỗi hàng
max_bytes_per_file = 25_000_000  # 25MB
rows_per_file = max_bytes_per_file // bytes_per_row  # ~7971 hàng

# Tổng số hàng và số file cần thiết
total_rows = X.shape[0]  # 70000
num_files = (total_rows + rows_per_file - 1) // rows_per_file  # Làm tròn lên

# Chia và lưu X thành nhiều file
for i in range(num_files):
    start_idx = i * rows_per_file
    end_idx = min((i + 1) * rows_per_file, total_rows)
    X_chunk = X[start_idx:end_idx]
    np.save(f"X_part_{i}.npy", X_chunk)
    print(f"Đã lưu: X_part_{i}.npy với {X_chunk.shape[0]} hàng, kích thước ~{X_chunk.nbytes / 1_000_000:.2f}MB")

# Lưu y nguyên bản
np.save("y.npy", y)
print("Đã lưu: y.npy")

print(f"Tổng cộng {num_files} file X_part_*.npy được tạo.")