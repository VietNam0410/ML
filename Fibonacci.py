import streamlit as st

def main():
    st.title("Tính dãy Fibonacci 🎯")

    # Người dùng nhập số n
    n = st.number_input("Nhập số nguyên dương n", min_value=0, step=1, value=10)

    # Hàm tính dãy Fibonacci
    def fibonacci(n):
        fib_series = [0, 1]
        for i in range(2, n):
            fib_series.append(fib_series[-1] + fib_series[-2])
        return fib_series[:n]

    if st.button("Tính dãy Fibonacci"):
        result = fibonacci(n)
        st.success(f"Dãy Fibonacci ({n} số đầu tiên): {result}")

