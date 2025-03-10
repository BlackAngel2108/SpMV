# **Sparse Matrix-Vector Multiplication (SpMV)**

This project implements **Sparse Matrix-Vector Multiplication (SpMV)** using six different sparse matrix storage formats. The goal is to provide a flexible and efficient solution for multiplying a sparse matrix by a dense vector, which is a common operation in scientific computing, machine learning, and graph processing.

The project is implemented in **C++**.

## **Features**

- **Efficient Multiplication**: Optimized for performance with various storage formats.
- **Extensible Design**: Easily add new storage formats or modify existing ones.
- **Comprehensive Testing**: Includes unit tests to ensure the correctness of each format.

## **Supported Storage Formats**

The following sparse matrix storage formats are supported:

1. **COO (Coordinate Format)**: Stores the matrix as a list of tuples (row, column, value).
2. **CSR (Compressed Sparse Row)**: Stores the matrix using three arrays: row pointers, column indices, and values.
3. **DIA (Diagonal Format)**: Efficient for matrices with a small number of non-zero diagonals.
4. **ELL (ELLPACK Format)**: Suitable for matrices with a consistent number of non-zeros per row.
5. **SELL_C**: An extension of the ELL format that includes additional storage for handling variable numbers of non-zeros per row. This format is efficient for matrices with a varying number of non-zeros across rows.
6. **SELL_C_sigma**: A variant of SELL_C that incorporates a permutation strategy to improve cache utilization and memory access patterns. This format is designed to enhance performance on modern architectures by reducing cache misses and improving data locality.

