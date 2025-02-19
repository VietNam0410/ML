# #Fatorial Calculator
# def factorial(n):
#     if n == 0 or n == 1:
#         return 1
#     else:
#         return n * factorial(n-1)

import streamlit as st

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def main():
    st.header("📌 Bài tập tính giai thừa")
    number = st.number_input("Nhập số nguyên", min_value=0, max_value=100, value=5)
    if st.button("Tính toán"):
        result = factorial(number)
        st.success(f"Giai thừa của {number}! là {result}")
if __name__ == "__main__":
    main()
